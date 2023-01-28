from mesa.batchrunner import BatchRunnerMP, batch_run
from model import MinimalModel
import multiprocessing

parameters = {
    # "magnitude": [5, 6, 7],
    "num_citizens": [3000],
    "num_buildings": [100],
    "num_hospitals": [5],
    "num_ambulances": [0, 1, 5, 15],
    "num_doctors": [1],
    "dispatch_size": [5, 10]
}

'Define the parameter settings to be tested'

model_reporters = {"num_citizens": lambda m: m.schedule.get_agent_count(),
                   # "num_buildings": lambda m: m.schedule.get_agent_count(),
                   # "num_hospitals": lambda m: m.schedule.get_agent_count(),
                   # "num_ambulances": lambda m: m.schedule.get_agent_count(),
                   # "num_doctors": lambda m: m.schedule.get_agent_count(),
                   # "dispatcher_size": lambda m: m.dispatcher.size,
                   # "num_dead_citizens": lambda m: m.dead_citizens,
                   # "num_saved_citizens": lambda m: m.saved_citizens,
                   # "num_injured_citizens": lambda m: m.injured_citizens,
                   # "num_dead_doctors": lambda m: m.dead_doctors,
                   # "num_saved_doctors": lambda m: m.saved_doctors,
                   # "num_injured_doctors": lambda m: m.injured_doctors,
                   # "num_destroyed_buildings": lambda m: m.destroyed_buildings,
                   # "num_damaged_buildings": lambda m: m.damaged_buildings
                   }

'keep track of the agents position and the state of the city'

""" class BatchrunnerExtender(BatchRunnerMP):
        def _result_prep_mp(self, results):
            
            Helper Function
            :param results: Takes results dictionary from Processpool and single processor debug run and fixes format to
         make compatible with BatchRunner Output
         :updates model_vars and agents_vars so consistent across all batchrunner
         
            # Take results and convert to dictionary so dataframe can be called
            for model_key, model in results.items():
                if self.model_reporters:
                    self.model_vars[model_key] = self.collect_model_vars(model)
                if self.agent_reporters:
                    agent_vars = self.collect_agent_vars(model)
                    for agent_id, reports in agent_vars.items():
                        agent_key = model_key + (agent_id,)
                        self.agent_vars[agent_key] = reports
                if hasattr(model, "datacollector"):
                    if model.datacollector.model_reporters is not None:
                        self.datacollector_model_reporters[
                            model_key
                        ] = model.datacollector.get_model_vars_dataframe()
                    if model.datacollector.agent_reporters is not None:
                        self.datacollector_agent_reporters[
                            model_key
                        ] = model.datacollector.get_agent_vars_dataframe('Citizen')

        # Make results consistent
            if len(self.datacollector_model_reporters.keys()) == 0:
                self.datacollector_model_reporters = None
            if len(self.datacollector_agent_reporters.keys()) == 0:
                self.datacollector_agent_reporters = None """

# def run_iteration(self, kwargs, param_values, run_count):
#     """ Run one iteration of the model, with the given parameters. """
#     model = self.model_cls(**kwargs)
#     results = self.run_model(model)
#     if param_values is not None:
#         model_key = tuple(param_values) + (run_count,)
#     else:
#         model_key = (run_count,)
#
#     if self.model_reporters:
#         self.model_vars[model_key] = self.collect_model_vars(model)
#     if self.agent_reporters:
#         agent_vars = self.collect_agent_vars(model)
#         for agent_id, reports in agent_vars.items():
#             agent_key = model_key + (agent_id,)
#             self.agent_vars[agent_key] = reports
#     # Collects data from datacollector object in model
#     if results is not None:
#         if results.model_reporters is not None:
#             self.datacollector_model_reporters[
#                 model_key
#             ] = results.get_model_vars_dataframe()
#         if results.agent_reporters is not None:
#             self.datacollector_agent_reporters[
#                 model_key
#             ] = results.get_agent_vars_dataframe('Citizen')

# batch_run = BatchrunnerExtender(MinimalModel, parameters, max_steps=5)

'specifying the model class, the parameter settings, the number of times to run each set of parameters, and the reporters to be tracked'

run = batch_run(MinimalModel, parameters, number_processes=1, max_steps= 10)
'run the Batchrunner'

# results = batch_run.get_model_vars_dataframe()

'Collect and analyze the results of the batch run to get the dataframe of the results'
