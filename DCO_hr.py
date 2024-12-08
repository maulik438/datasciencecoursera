import os
from json import load
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from datetime import timedelta
import math

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, vessels_file, ports_csv):
        self.vessels_df = pd.read_csv(vessels_file)
        self.ports_df = pd.read_csv(ports_csv)

        self.ports = ['Port 1', 'Port 2']
        self.berths_per_port = {port: list(range(1, 3)) for port in self.ports}
        self.num_ports = len(self.ports)
        self.num_berths_per_port = {port: len(berths) for port, berths in self.berths_per_port.items()}
        self.num_vessels = len(self.vessels_df)

        self.initialize_data()
        self.create_genetic_algorithm_components()

    def initialize_data(self):
        self.vessels_df["departure_date_time"] = pd.to_datetime(self.vessels_df.departure_date + " " + self.vessels_df.departure_time)
        self.vessels_df["arrival_port_1_date_time"] = pd.to_datetime(self.vessels_df.arrival_date_port_1 + " " + self.vessels_df.arrival_time_port_1)
        self.vessels_df["arrival_port_2_date_time"] = pd.to_datetime(self.vessels_df.arrival_date_port_2 + " " + self.vessels_df.arrival_time_port_2)

        self.vessels_df["departure_date"] = pd.to_datetime(self.vessels_df["departure_date"])

        self.vessels = self.vessels_df["vessel"].tolist()
        self.departure_dates = self.vessels_df.set_index("vessel")["departure_date"].to_dict()

        self.arrival_dates = {
            vessel: {
                "Port 1": arrival_date_time_port1,
                "Port 2": arrival_date_time_port2
            } for vessel, arrival_date_time_port1, arrival_date_time_port2 in zip(self.vessels_df["vessel"], self.vessels_df["arrival_port_1_date_time"], self.vessels_df["arrival_port_2_date_time"])
        }

        self.demurrage_costs = self.vessels_df.set_index("vessel")["demurrage_cost"].to_dict()
        self.metric_quantities = self.vessels_df.set_index("vessel")["metric_quantity"].to_dict()
        self.discharge_rates = self.vessels_df.set_index("vessel")["discharge_rate"].to_dict()

        self.berthing_durations = {v: self.metric_quantities[v] / self.discharge_rates[v] for v in self.vessels}

        self.ports = np.unique(self.ports_df["Port"])

    def create_genetic_algorithm_components(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        self.toolbox.register("attr_int", random.randint, 0, self.num_ports * max(self.num_berths_per_port.values()) - 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_int, n=self.num_vessels)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate)
    
    def decode_individual(self, individual):
        allocations = []
        for gene in individual:
            port_idx = gene // max(self.num_berths_per_port.values())
            berth_idx = gene % max(self.num_berths_per_port.values())
            allocations.append((self.ports[port_idx], berth_idx))
        return allocations

    def evaluate(self, individual):
        total_demurrage_cost = 0
        total_journey_duration = datetime(1, 1, 1, 0, 0)
        total_pre_berthing_delay = datetime(1, 1, 1, 0, 0)
        start_day_time = datetime(1, 1, 1, 0, 0)
        start_times = {port: [start_day_time] * self.num_berths_per_port[port] for port in self.ports}

        allocations = self.decode_individual(individual)
        berth_allocations = {port: [None] * self.num_berths_per_port[port] for port in self.ports}
        hr = timedelta(hours=1)

        for i, (port, berth_idx) in enumerate(allocations):
            vessel = self.vessels[i]
            best_start_time = max(self.arrival_dates[vessel][port], start_times[port][berth_idx])
            end_time = best_start_time + math.ceil(timedelta(days=self.berthing_durations[vessel]) / hr) + hr

            if berth_allocations[port][berth_idx] is not None:
                allocated_start, allocated_end = berth_allocations[port][berth_idx]
                if best_start_time < allocated_end:
                    return float('inf'),
            berth_allocations[port][berth_idx] = (best_start_time, end_time)

            pre_berthing_delay = best_start_time - self.arrival_dates[vessel][port]
            demurrage_cost = (pre_berthing_delay.days + (pre_berthing_delay.seconds / 3600 / 24)) * self.demurrage_costs[vessel]
            journey_duration = best_start_time - self.departure_dates[vessel]

        berth_allocations[port][berth_idx] = (best_start_time, end_time)

        start_times[port][berth_idx] = end_time
        total_pre_berthing_delay += pre_berthing_delay
        total_demurrage_cost += demurrage_cost
        total_journey_duration += journey_duration

        for port in self.ports:
            for berth_idx in range(self.num_berths_per_port[port]):
                allocations_for_berth = [(i, start_times[port][berth_idx]) for i, (p, b) in enumerate(allocations) if p == port and b == berth_idx]
                allocations_for_berth.sort(key=lambda x: x[1])

                for i in range(1, len(allocations_for_berth)):
                    prev_end = start_times[port][berth_idx]
                    current_start = allocations_for_berth[i][1]
                    if current_start < prev_end:
                        return float('inf'),

        total_cost = total_journey_duration.toordinal() + total_pre_berthing_delay.toordinal() + total_demurrage_cost
        return total_cost,

    def simulated_annealing(self, individual, initial_temp=100, cooling_rate=0.99, max_iter=100):
        current_ind = self.toolbox.clone(individual)
        current_fit = self.toolbox.evaluate(current_ind)
        best_ind = current_ind
        best_fit = current_fit
        temp = initial_temp

        for _ in range(max_iter):
            new_ind = self.toolbox.clone(current_ind)
            self.toolbox.mutate(new_ind)
            new_fit = self.toolbox.evaluate(new_ind)

            if sum(new_fit) < sum(current_fit) or random.uniform(0, 1) < np.exp((sum(current_fit) - sum(new_fit)) / temp):
                current_ind = new_ind
                current_fit = new_fit

            if sum(new_fit) < sum(best_fit):
                best_ind = new_ind
                best_fit = new_fit

            temp *= cooling_rate

        return best_ind

    def run(self, population_size=50, generations=100):  # Todo: gen 100
        population = self.toolbox.population(n=population_size)

        for gen in range(generations):
            print(f"Generation {gen}")
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.5, mutpb=0.2)
            fits = map(self.toolbox.evaluate, offspring)

            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            for i in range(len(offspring)):
                offspring[i] = self.simulated_annealing(offspring[i])

            population = self.toolbox.select(offspring + population, k=len(population))

        best_ind = tools.selBest(population, 1)[0]
        self.best_schedule = self.decode_individual(best_ind)

        self.best_schedule = best_ind
        return best_ind

    def generate_insights(self):
        hr = timedelta(hours=1)
        insights = []

        start_times = {port: [datetime(1, 1, 1, 0, 0)] * self.num_berths_per_port[port] for port in self.ports}

        allocations = self.decode_individual(self.best_schedule)

        for i, (port, berth_idx) in enumerate(allocations):
            vessel = self.vessels[i]
            start_time = max(self.arrival_dates[vessel][port], start_times[port][berth_idx])
            end_time = start_time + math.ceil(timedelta(days=self.berthing_durations[vessel]) / hr) * hr

            pre_berthing_delay = start_time - self.arrival_dates[vessel][port]
            demurrage_cost = self.demurrage_costs[vessel] * (pre_berthing_delay.days + (pre_berthing_delay.seconds / 3600 / 24))

            journey_duration = start_time - self.departure_dates[vessel]
            if journey_duration.days < 0:
                journey_duration = timedelta()

            insights.append({
                'vessel': vessel,
                'port': port,
                'berth': berth_idx + 1,
                'start time': start_time,
                'end time': end_time,
                'pre berthing delay': pre_berthing_delay,
                'demurrage cost': demurrage_cost,
                'journey duration': journey_duration
            })

            start_times[port][berth_idx] = end_time

        insights_df = pd.DataFrame(insights)
        print(insights_df)
        return insights_df

# We are doing Visualization of optimal berth allocation with time-space graph
fig, ax = plt.subplots(figsize=(14, 7))
colors = list(mcolors.TABLEAU_COLORS.keys())

def get_visual(insights_df):
    for idx, row in insights_df.iterrows():
        vessel = row['vessel']
        port = row['port']
        berth = row['berth']
        start_datetime = row['start time']
        end_datetime = row['end time']
        pbd = row['pre berthing delay']
        demurrage_cost = row['demurrage cost']
        berth_label = f"{port} berth {berth}"
        color = colors[idx % len(colors)]

        ax.barh(y=berth_label, width=(end_datetime - start_datetime).days, left=start_datetime, color=color)
        ax.text(start_datetime + (end_datetime - start_datetime) / 2, berth_label,
                f"{vessel}\nStart: {start_datetime.strftime('%Y-%m-%d %H:%M')}\nEnd: {end_datetime.strftime('%Y-%m-%d %H:%M')}\n"
                f"PBD: {str(pbd)[:-3]} hrs \n Demurrage Cost: {demurrage_cost:.2f}$",
                horizontalalignment='center', verticalalignment='center', fontsize=8, color='black')

    ax.set_xlabel('Time')
    ax.set_ylabel('Berths')
    ax.set_title('Optimal Berth Allocation Gantt Chart')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.savefig(r'../output/berth_allocation_GA_hrs_1.jpg')
    plt.show()
    plt.close(fig) #close the figure window
    
    # We are doing Visualization of optimal berth allocation with time-space graph
plt.subplots(figsize=(14, 7))
colors = list(mcolors.TABLEAU_COLORS.keys())

def visual(insights_df):
    for idx, row in insights_df.iterrows():
        vessel = row['vessel']
        port = row['port']
        berth = row['berth']
        start_datetime = row['start time']
        end_datetime = row['end time']
        pbd = row['pre berthing delay']
        demurrage_cost = row['demurrage cost']
        berth_label = f"{port} berth {berth}"
        color = colors[idx % len(colors)]

        ax.barh(y=berth_label, width=(end_datetime - start_datetime).days, left=start_datetime, color=color)
        ax.text(start_datetime + (end_datetime - start_datetime) / 2, berth_label,
                f"{vessel}\nStart: {start_datetime.strftime('%Y-%m-%d %H:%M')}\nEnd: {end_datetime.strftime('%Y-%m-%d %H:%M')}\n"
                f"PBD: {str(pbd)[:-3]} hrs \n Demurrage Cost: {demurrage_cost:.2f}$",
                horizontalalignment='center', verticalalignment='center', fontsize=8, color='black')

    ax.set_xlabel('Time')
    ax.set_ylabel('Berths')
    ax.set_title('Optimal Berth Allocation Gantt Chart')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.savefig(r'../output/berth_allocation_GA_hrs_1.jpg')
    plt.show()
    plt.close(fig) #close the figure window

def get_params():
    file_name = "params.json"
    file_path = os.path.join(r'../lib', file_name)
    logger.info('file_path: {}'.format(file_path))
    with open(file_path) as f:
        params = load(f)
    return params
