
import json
import carla

with open('./env/CARLA_0_9_13_pithy/weather.json', 'r', encoding='utf8') as fff:
    weather_params = json.load(fff)
with open('./env/CARLA_0_9_13_pithy/scenario.json', 'r', encoding='utf8') as fff:
    scenario_params = json.load(fff)

selected_scenario = "oppo-high-beam"


# client init
client = carla.Client('localhost', 12345)
client.set_timeout(30)

# world
world = client.load_world(scenario_params[selected_scenario]["map"])

# blueprint
bp_lib = world.get_blueprint_library()

# manager
tm = client.get_trafficmanager(19456)
lm = world.get_lightmanager()