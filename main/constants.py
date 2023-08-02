from typing import Tuple

########################################################################################################################
# Special objects
########################################################################################################################
BULKY_OBJECTS = [
    "first countertop",
    "second countertop",
    "third countertop",
    "fourth countertop",
    "first stove burner",
    "second stove burner",
    "third stove burner",
    "fourth stove burner",  
    "faucet",  
    "sink basin",
    "sink",
    "window",
    "wall",
    "drawer",
    "cabinet",
    "fridge",
    "coffee machine"
]

NAME_MAP = {
    "TomatoSliced": "tomato slice",
    "PotatoSliced": "potato slice",
    "LettuceSliced": "lettuce slice",
    "BreadSliced": "bread slice",
    "EggCracked": "cracked egg",
    "AppleSliced": "apple slice",
    "HousePlant": "house plant",
    "CounterTop-1": "first countertop",
    "CounterTop-2": "second countertop",
    "CounterTop-3": "third countertop",
    "CounterTop-4": "fourth countertop",
    "StoveBurner-1": "first stove burner",
    "StoveBurner-2": "second stove burner",
    "StoveBurner-3": "third stove burner",
    "StoveBurner-4": "fourth stove burner",
    "Cabinet-1": "first cabinet",
    "Cabinet-2": "second cabinet",
    "Cabinet-3": "third cabinet",
    "Cabinet-4": "fourth cabinet",
    "Cabinet-5": "fifth cabinet",
    "Cabinet-6": "sixth cabinet",
    "Cabinet-7": "seventh cabinet",
    "Cabinet-8": "eight cabinet",
    "Cabinet-9": "ninth cabinet",
    "Cabinet-10": "tenth cabinet",
    "Cabinet-11": "eleventh cabinet",
    "Cabinet-12": "twelveth cabinet",
    "Faucet-1": "first faucet",
    "Faucet-2": "second faucet",
    "Sink-1": "first sink",
    "Sink-2": "second sink",
    "StoveBurner": "stove burner",
    'AlarmClock': "alarm clock",
    'BaseballBat': "baseball bat",
    'BathtubBasin': "bathtub basin",
    'ButterKnife': "butter knife",
    'CoffeeMachine': "coffee machine",
    'CreditCard': "credit card",
    'DeskLamp': "desk lamp",
    'DishSponge': "dish sponge",
    'FloorLamp': "floor lamp",
    'GarbageCan': "garbage can",
    'Glassbottle': "glass bottle",
    'HandTowel': "hand towel",
    'HandTowelHolder': "towel holder",
    'LaundryHamper': "laundry hamper",
    'LaundryHamperLid': "laundry hamper lid",
    'LightSwitch': "light switch",
    'PaperTowel': "paper towel",
    'PaperTowelRoll': "paper towel roll",
    'PepperShaker': "pepper shaker",
    'RemoteControl': "remote control",
    'SaltShaker': "salt shaker",
    'ScrubBrush': "scrub brush",
    'ShowerDoor': "shower door",
    'ShowerGlass': "shower glass",
    'SinkBasin': "sink",
    'SoapBar': "soap bar",
    'SoapBottle': "soap bottle",
    'SprayBottle': "spray bottle",
    'StoveKnob': "stove knob",
    'DiningTable': "dining table",
    'CoffeeTable': "coffee table",
    'SideTable': "side table",
    'TennisRacket': "tennis racket",
    'TissueBox': "tissue box",
    'ToiletPaper': "toilet paper",
    'ToiletPaperHanger': "toilet paper holder",
    'ToiletPaperRoll': "toilet paper roll",
    'TowelHolder': "towel holder",
    'TVStand': "tv stand",
    'WateringCan': "watering can",
    'WineBottle': "wine bottle",
    "ShelvingUnit": "shelving unit",
}

OBJ_SLICED_MAP = {
    "Tomato": "TomatoSliced",
    "Potato": "PotatoSliced",
    "Lettuce": "LettuceSliced",
    "Bread": "BreadSliced",
    "Egg": "EggCracked",
    "Apple": "AppleSliced"
}

OBJ_UNSLICED_MAP = {
    "TomatoSliced": "Tomato",
    "PotatoSliced": "Potato",
    "LettuceSliced": "Lettuce",
    "BreadSliced": "Bread",
    "AppleSliced": "Apple",
    "EggCracked": "Egg"
}

SOUND_PATH = {
    "Toggle on StoveBurner": "toggle-on-stoveburner.wav",
    "Toggle on Toaster": "toggle-on-toaster.wav",
    "Toggle on Faucet": "toggle-on-faucet.wav",
    "Toggle on Television": "toggle-on-television.wav",
    "Open Fridge": "open-fridge.wav",
    "Close Fridge": "close-fridge.wav",
    "Close Laptop": "close-laptop.wav",
    "Drop Knife": "drop-knife.wav",
    "Drop Bowl": "drop-plastic-bowl.wav",
    "Drop RemoteControl": "drop-plastic-bowl.wav",
    "Drop Egg": "drop-egg.wav",
    "Drop Pot": "drop-pot.wav",
    "Drop Mug": "drop-plastic-bowl.wav",
    "Drop Plate": "drop-plate.wav",
    "Slice Bread": "slice-bread.wav",
    "Crack Egg": "crack-egg.wav",
    "Open Microwave": "open-microwave.wav",
    "Close Microwave": "close-microwave.wav",
    "Toggle on Microwave": "toggle-on-microwave.wav",
    "Toggle on CoffeeMachine": "toggle-on-coffeemachine.wav",
    "Pour water into Mug": "pour-water-in-mug.wav",
    "Pour water into Sink": "pour-water-in-sink.wav",
}

TASK_DICT = {
    0: '',
    1: 'boilWater',
    2: 'toastBread',
    3: 'cookEgg',
    4: 'heatPotato',
    5: 'makeCoffee',
    6: 'waterPlant',
    7: 'storeEgg',
    8: 'makeSalad',
    9: 'switchDevices',
    10: 'warmWater'
}

########################################################################################################################
# Unity Hyperparameters
########################################################################################################################

BUILD_PATH = None
X_DISPLAY = '0'

AGENT_STEP_SIZE = 0.25
AGENT_HORIZON_ADJ = 15
AGENT_ROTATE_ADJ = 90
CAMERA_HEIGHT_OFFSET = 0.75
VISIBILITY_DISTANCE = 1.5
HORIZON_GRANULARITY = 15

RENDER_IMAGE = True
RENDER_DEPTH_IMAGE = True
RENDER_CLASS_IMAGE = True
RENDER_OBJECT_IMAGE = True

MAX_DEPTH = 5000
STEPS_AHEAD = 5
SCENE_PADDING = STEPS_AHEAD * 3
SCREEN_WIDTH = DETECTION_SCREEN_WIDTH = 300
SCREEN_HEIGHT = DETECTION_SCREEN_HEIGHT = 300
MIN_VISIBLE_PIXELS = 10

# (400) / (600*600) ~ 0.13% area of image
# int(MIN_VISIBLE_RATIO * float(DETECTION_SCREEN_WIDTH) * float(DETECTION_SCREEN_HEIGHT))
# MIN_VISIBLE_PIXELS = int(MIN_VISIBLE_RATIO * float(DETECTION_SCREEN_WIDTH) * float(
#    DETECTION_SCREEN_HEIGHT))  # (400) / (600*600) ~ 0.13% area of image

########################################################################################################################
# Scenes and Objects
########################################################################################################################

TRAIN_SCENE_NUMBERS = list(range(7, 31))           # Train Kitchens (24/30)
TRAIN_SCENE_NUMBERS.extend(list(range(207, 231)))  # Train Living Rooms (24/30)
TRAIN_SCENE_NUMBERS.extend(list(range(307, 331)))  # Train Bedrooms (24/30)
TRAIN_SCENE_NUMBERS.extend(list(range(407, 431)))  # Train Bathrooms (24/30)

TEST_SCENE_NUMBERS = list(range(1, 7))             # Test Kitchens (6/30)
TEST_SCENE_NUMBERS.extend(list(range(201, 207)))   # Test Living Rooms (6/30)
TEST_SCENE_NUMBERS.extend(list(range(301, 307)))   # Test Bedrooms (6/30)
TEST_SCENE_NUMBERS.extend(list(range(401, 407)))   # Test Bathrooms (6/30)

SCENE_NUMBERS = TRAIN_SCENE_NUMBERS + TEST_SCENE_NUMBERS

# Scene types.
SCENE_TYPE = {"Kitchen": range(1, 31),
              "LivingRoom": range(201, 231),
              "Bedroom": range(301, 331),
              "Bathroom": range(401, 431)}

OBJECTS = [
    'AlarmClock',
    'Apple',
    'ArmChair',
    'BaseballBat',
    'BasketBall',
    'Bathtub',
    'BathtubBasin',
    'Bed',
    'Blinds',
    'Book',
    'Boots',
    'Bowl',
    'Box',
    'Bread',
    'ButterKnife',
    'Cabinet',
    'Candle',
    'Cart',
    'CD',
    'CellPhone',
    'Chair',
    'Cloth',
    'CoffeeMachine',
    'CounterTop',
    'CreditCard',
    'Cup',
    'Curtains',
    'Desk',
    'DeskLamp',
    'DishSponge',
    'Drawer',
    'Dresser',
    'Egg',
    'FloorLamp',
    'Footstool',
    'Fork',
    'Fridge',
    'GarbageCan',
    'Glassbottle',
    'HandTowel',
    'HandTowelHolder',
    'HousePlant',
    'Kettle',
    'KeyChain',
    'Knife',
    'Ladle',
    'Laptop',
    'LaundryHamper',
    'LaundryHamperLid',
    'Lettuce',
    'LightSwitch',
    'Microwave',
    'Mirror',
    'Mug',
    'Newspaper',
    'Ottoman',
    'Painting',
    'Pan',
    'PaperTowel',
    'PaperTowelRoll',
    'Pen',
    'Pencil',
    'PepperShaker',
    'Pillow',
    'Plate',
    'Plunger',
    'Poster',
    'Pot',
    'Potato',
    'RemoteControl',
    'Safe',
    'SaltShaker',
    'ScrubBrush',
    'Shelf',
    'ShowerDoor',
    'ShowerGlass',
    'Sink',
    'SinkBasin',
    'SoapBar',
    'SoapBottle',
    'Sofa',
    'Spatula',
    'Spoon',
    'SprayBottle',
    'Statue',
    'StoveBurner',
    'StoveKnob',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'TeddyBear',
    'Television',
    'TennisRacket',
    'TissueBox',
    'Toaster',
    'Toilet',
    'ToiletPaper',
    'ToiletPaperHanger',
    'ToiletPaperRoll',
    'Tomato',
    'Towel',
    'TowelHolder',
    'TVStand',
    'Vase',
    'Watch',
    'WateringCan',
    'Window',
    'WineBottle',
]

OBJECTS_LOWER_TO_UPPER = {obj.lower(): obj for obj in OBJECTS}

OBJECTS_SINGULAR = [
    'alarmclock',
    'apple',
    'armchair',
    'baseballbat',
    'basketball',
    'bathtub',
    'bathtubbasin',
    'bed',
    'blinds',
    'book',
    'boots',
    'bowl',
    'box',
    'bread',
    'butterknife',
    'cabinet',
    'candle',
    'cart',
    'cd',
    'cellphone',
    'chair',
    'cloth',
    'coffeemachine',
    'countertop',
    'creditcard',
    'cup',
    'curtains',
    'desk',
    'desklamp',
    'dishsponge',
    'drawer',
    'dresser',
    'egg',
    'floorlamp',
    'footstool',
    'fork',
    'fridge',
    'garbagecan',
    'glassbottle',
    'handtowel',
    'handtowelholder',
    'houseplant',
    'kettle',
    'keychain',
    'knife',
    'ladle',
    'laptop',
    'laundryhamper',
    'laundryhamperlid',
    'lettuce',
    'lightswitch',
    'microwave',
    'mirror',
    'mug',
    'newspaper',
    'ottoman',
    'painting',
    'pan',
    'papertowel',
    'papertowelroll',
    'pen',
    'pencil',
    'peppershaker',
    'pillow',
    'plate',
    'plunger',
    'poster',
    'pot',
    'potato',
    'remotecontrol',
    'safe',
    'saltshaker',
    'scrubbrush',
    'shelf',
    'showerdoor',
    'showerglass',
    'sink',
    'sinkbasin',
    'soapbar',
    'soapbottle',
    'sofa',
    'spatula',
    'spoon',
    'spraybottle',
    'statue',
    'stoveburner',
    'stoveknob',
    'diningtable',
    'coffeetable',
    'sidetable'
    'teddybear',
    'television',
    'tennisracket',
    'tissuebox',
    'toaster',
    'toilet',
    'toiletpaper',
    'toiletpaperhanger',
    'toiletpaperroll',
    'tomato',
    'towel',
    'towelholder',
    'tvstand',
    'vase',
    'watch',
    'wateringcan',
    'window',
    'winebottle',
]

OBJECTS_PLURAL = [
    'alarmclocks',
    'apples',
    'armchairs',
    'baseballbats',
    'basketballs',
    'bathtubs',
    'bathtubbasins',
    'beds',
    'blinds',
    'books',
    'boots',
    'bottles',
    'bowls',
    'boxes',
    'bread',
    'butterknives',
    'cabinets',
    'candles',
    'carts',
    'cds',
    'cellphones',
    'chairs',
    'cloths',
    'coffeemachines',
    'countertops',
    'creditcards',
    'cups',
    'curtains',
    'desks',
    'desklamps',
    'dishsponges',
    'drawers',
    'dressers',
    'eggs',
    'floorlamps',
    'footstools',
    'forks',
    'fridges',
    'garbagecans',
    'glassbottles',
    'handtowels',
    'handtowelholders',
    'houseplants',
    'kettles',
    'keychains',
    'knives',
    'ladles',
    'laptops',
    'laundryhampers',
    'laundryhamperlids',
    'lettuces',
    'lightswitches',
    'microwaves',
    'mirrors',
    'mugs',
    'newspapers',
    'ottomans',
    'paintings',
    'pans',
    'papertowels',
    'papertowelrolls',
    'pens',
    'pencils',
    'peppershakers',
    'pillows',
    'plates',
    'plungers',
    'posters',
    'pots',
    'potatoes',
    'remotecontrollers',
    'safes',
    'saltshakers',
    'scrubbrushes',
    'shelves',
    'showerdoors',
    'showerglassess',
    'sinks',
    'sinkbasins',
    'soapbars',
    'soapbottles',
    'sofas',
    'spatulas',
    'spoons',
    'spraybottles',
    'statues',
    'stoveburners',
    'stoveknobs',
    'diningtables',
    'coffeetables',
    'sidetable',
    'teddybears',
    'televisions',
    'tennisrackets',
    'tissueboxes',
    'toasters',
    'toilets',
    'toiletpapers',
    'toiletpaperhangers',
    'toiletpaperrolls',
    'tomatoes',
    'towels',
    'towelholders',
    'tvstands',
    'vases',
    'watches',
    'wateringcans',
    'windows',
    'winebottles',
]

_RECEPTACLE_OBJECTS = [
    'BathtubBasin',
    'Bowl',
    'Cup',
    'Drawer',
    'Mug',
    'Plate',
    'Shelf',
    'Sink',
    'SinkBasin',
    'Box',
    'Cabinet',
    'CoffeeMachine',
    'CounterTop',
    'Fridge',
    'GarbageCan',
    'HandTowelHolder',
    'Microwave',
    'PaintingHanger',
    'Pan',
    'Pot',
    'StoveBurner',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'ToiletPaperHanger',
    'TowelHolder',
    'Safe',
    'BathtubBasin',
    'ArmChair',
    'Toilet',
    'Sofa',
    'Ottoman',
    'Dresser',
    'LaundryHamper',
    'Desk',
    'Bed',
    'Cart',
    'TVStand',
    'Toaster',
]

_MOVABLE_RECEPTACLES = [
    'Bowl',
    'Box',
    'Cup',
    'Mug',
    'Plate',
    'Pan',
    'Pot',
]

_INTERACTIVE_OBJECTS = [
    'AlarmClock',
    'Apple',
    'ArmChair',
    'BaseballBat',
    'BasketBall',
    'Bathtub',
    'BathtubBasin',
    'Bed',
    'Blinds',
    'Book',
    'Boots',
    'Bowl',
    'Box',
    'Bread',
    'BreadSliced',
    'ButterKnife',
    'Cabinet',
    'Candle',
    'Cart',
    'CD',
    'CellPhone',
    'Chair',
    'Cloth',
    'CoffeeMachine',
    'CounterTop',
    'CreditCard',
    'Cup',
    'Curtains',
    'Desk',
    'DeskLamp',
    'DishSponge',
    'Drawer',
    'Dresser',
    'Egg',
    'EggCracked',
    'FloorLamp',
    'Footstool',
    'Fork',
    'Fridge',
    'GarbageCan',
    'Glassbottle',
    'HandTowel',
    'HandTowelHolder',
    'HousePlant',
    'Kettle',
    'KeyChain',
    'Knife',
    'Ladle',
    'Laptop',
    'LaundryHamper',
    'LaundryHamperLid',
    'Lettuce',
    'LettuceSliced',
    'LightSwitch',
    'Microwave',
    'Mirror',
    'Mug',
    'Newspaper',
    'Ottoman',
    'Painting',
    'Pan',
    'PaperTowel',
    'PaperTowelRoll',
    'Pen',
    'Pencil',
    'PepperShaker',
    'Pillow',
    'Plate',
    'Plunger',
    'Poster',
    'Pot',
    'Potato',
    'PotatoSliced',
    'RemoteControl',
    'Safe',
    'SaltShaker',
    'ScrubBrush',
    'Shelf',
    'ShowerDoor',
    'ShowerGlass',
    'Sink',
    'SinkBasin',
    'SoapBar',
    'SoapBottle',
    'Sofa',
    'Spatula',
    'Spoon',
    'SprayBottle',
    'Statue',
    'StoveBurner',
    'StoveKnob',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'TeddyBear',
    'Television',
    'TennisRacket',
    'TissueBox',
    'Toaster',
    'Toilet',
    'ToiletPaper',
    'ToiletPaperHanger',
    'ToiletPaperRoll',
    'Tomato',
    'TomatoSliced',
    'Towel',
    'TowelHolder',
    'TVStand',
    'Vase',
    'Watch',
    'WateringCan',
    'Window',
    'WineBottle',
]

_TABLETOP_OBJECTS = [
    'AlarmClock',
    'Apple',
    'BaseballBat',
    'BasketBall',
    'Book',
    'Boots',
    'Bowl',
    'Box',
    'Bread',
    'ButterKnife',
    'Cabinet',
    'Candle',
    'Cart',
    'CD',
    'CellPhone',
    'Cloth',
    'CoffeeMachine',
    'CreditCard',
    'Cup',
    'DeskLamp',
    'DishSponge',
    'Egg',
    'FloorLamp',
    'Footstool',
    'Fork',
    'GarbageCan',
    'Glassbottle',
    'HandTowel',
    'HandTowelHolder',
    'Kettle',
    'KeyChain',
    'Knife',
    'Ladle',
    'Laptop',
    'Lettuce',
    'Microwave',
    'Mug',
    'Newspaper',
    'Pan',
    'PaperTowel',
    'PaperTowelRoll',
    'Pen',
    'Pencil',
    'PepperShaker',
    'Pillow',
    'Plate',
    'Plunger',
    'Pot',
    'Potato',
    'RemoteControl',
    'Safe',
    'SaltShaker',
    'ScrubBrush',
    'Sink',
    'SinkBasin',
    'SoapBar',
    'SoapBottle',
    'Spatula',
    'Spoon',
    'SprayBottle',
    'Statue',
    'TeddyBear',
    'TennisRacket',
    'TissueBox',
    'Toaster',
    'ToiletPaper',
    'ToiletPaperHanger',
    'ToiletPaperRoll',
    'Tomato',
    'Towel',
    'TowelHolder',
    'Vase',
    'Watch',
    'WateringCan',
    'WineBottle',
]

_STRUCTURAL_OBJECTS = [
    "Books",
    "Ceiling",
    "Door",
    "Floor",
    "KitchenIsland",
    "LightFixture",
    "Rug",
    "Wall",
    "StandardWallSize",
    "Faucet",
    "Bottle",
    "Bag",
    "Cube",
    "Room",
]

MOVABLE_RECEPTACLES_SET = set(_MOVABLE_RECEPTACLES)
OBJECTS_SET = set(OBJECTS) | MOVABLE_RECEPTACLES_SET

OBJECT_CLASS_TO_ID = {obj: ii for (ii, obj) in enumerate(OBJECTS)}

_OPENABLES = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box', 'Kettle', 'Toilet', 'Laptop']

_SLICEABLES = [
    "Apple", 
    "Bread",
    "Egg"
    "Lettuce",
    "Potato",
    "Tomato",
]

_FILLABLE = [
    "Bottle",
    "Bowl",
    "Cup",
    "HousePlant",
    "Kettle",
    "Mug",
    "Pot",
    "Sink",
    "WateringCan",
    "WineBottle",
]

_TOGGLABLES = [
    "DeskLamp", 
    "FloorLamp",
    "Microwave", 
    "Candle", 
    "CellPhone", 
    "CoffeeMachine", 
    "Faucet", 
    "Laptop", 
    "LightSwitch",
    "ShowerHead",
    "StoveBurner",
    "StoveKnob",
    "Television",
    "Toaster"]

_DIRTYABLE = [
    "Lettuce",
    "LettuceSliced",
    "Tomato",
    "TomatoSliced",
    "Potato",
    "PotatoSliced",
    "Bread",
    "BreadSliced"
]

_BREAKABLE = [
    "Bottle",
    "Bowl",
    "CellPhone",
    "Cup",
    "Egg",
    "Laptop",
    "Mirror",
    "Mug",
    "Plate",
    "ShowerDoor",
    "ShowerGlass",
    "Statue",
    "Television",
    "Vase",
    "Window",
    "WineBottle"
]

_CRACKABLE = [
    "Egg"
]

_FLAT_RECEPT = [
    "CounterTop",
    "StoveBurner",
    "TVStand",
    "DiningTable",
    "SinkBasin"
]

INVENTORY_OBJECT_STR = "<InventoryObject>"
_EXTRA_TOKENS = [
    INVENTORY_OBJECT_STR
]

_PICKABLES = [s for s in _INTERACTIVE_OBJECTS if (
        (s not in _RECEPTACLE_OBJECTS) and
        (s not in _OPENABLES) and
        (s not in _TOGGLABLES)) or (s in _MOVABLE_RECEPTACLES)
    ]

OBJECT_CLASSES = _STRUCTURAL_OBJECTS + _INTERACTIVE_OBJECTS + _EXTRA_TOKENS

# Mappings between integers and strings
OBJECT_INT_TO_STR = {i: o for i, o in enumerate(OBJECT_CLASSES)}
OBJECT_STR_TO_INT = {o: i for i, o in enumerate(OBJECT_CLASSES)}
UNK_OBJ_INT = len(OBJECT_CLASSES)
UNK_OBJ_STR = "Unknown"
#COLOR_OTHERS = (255, 0, 0)
COLOR_OTHERS = (100, 100, 100)

# -------------------------------------------------------------
# Public API:
# -------------------------------------------------------------
# Simple mappings

def get_all_interactive_objects():
    return list(iter(_INTERACTIVE_OBJECTS))

def get_receptacle_ids():
    return [object_string_to_intid(s) for s in _RECEPTACLE_OBJECTS]

def get_pickable_ids():
    return [object_string_to_intid(s) for s in _PICKABLES]

def get_togglable_ids():
    return [object_string_to_intid(s) for s in _TOGGLABLES]

def get_openable_ids():
    return [object_string_to_intid(s) for s in _OPENABLES]

def get_sliceable_ids():
    return [object_string_to_intid(s) for s in _SLICEABLES]

def get_ground_ids():
    return [object_string_to_intid(s) for s in ["Rug", "Floor"]]

def get_num_objects():
    return len(OBJECT_CLASSES) + 1

def object_color_to_intid(color: Tuple[int, int, int]) -> int:
    global OBJECT_COLOR_TO_INTID
    return OBJECT_COLOR_TO_INTID[color]

def object_intid_to_color(intid: int) -> Tuple[int, int, int]:
    global OBJECT_INTID_TO_COLOR
    return OBJECT_INTID_TO_COLOR[intid]

def object_string_to_intid(object_str) -> int:
    global OBJECT_STR_TO_INT, UNK_OBJ_INT
    # Remove the part about object instance location
    object_str = object_str.split("|")[0].split(":")[-1].split(".")[0]
    if object_str in OBJECT_STR_TO_INT:
        return OBJECT_STR_TO_INT[object_str]
    else:
        return UNK_OBJ_INT

INTERACTIVE_OBJECT_IDS = [object_string_to_intid(o) for o in _INTERACTIVE_OBJECTS]
STRUCTURAL_OBJECT_IDS = [object_string_to_intid(o) for o in _STRUCTURAL_OBJECTS]
TABLETOP_OBJECT_IDS = [object_string_to_intid(o) for o in _TABLETOP_OBJECTS]

def object_intid_to_string(intid: int) -> str:
    global OBJECT_INT_TO_STR, UNK_OBJ_STR
    if intid in OBJECT_INT_TO_STR:
        return OBJECT_INT_TO_STR[intid]
    else:
        return UNK_OBJ_STR

def object_string_to_color(object_str : str) -> Tuple[int, int, int]:
    return object_intid_to_color(object_string_to_intid((object_str)))

def object_color_to_string(color: Tuple[int, int, int]) -> str:
    return object_intid_to_string(object_color_to_intid(color))

# object parents
OBJ_PARENTS = {obj: obj for obj in OBJECTS}
OBJ_PARENTS['AppleSliced'] = 'Apple'
OBJ_PARENTS['BreadSliced'] = 'Bread'
OBJ_PARENTS['LettuceSliced'] = 'Lettuce'
OBJ_PARENTS['PotatoSliced'] = 'Potato'
OBJ_PARENTS['TomatoSliced'] = 'Tomato'

# force a different horizon view for objects of (type, location). If the location is None, force this horizon for all
# objects of that type.
FORCED_HORIZON_OBJS = {
    ('FloorLamp', None): 0,
    ('Fridge', 18): 30,
    ('Toilet', None): 15,
}

# openable objects with fixed states for transport.
FORCED_OPEN_STATE_ON_PICKUP = {
    'Laptop': False,
}

# list of openable classes.
OPENABLE_CLASS_LIST = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box']
OPENABLE_CLASS_SET = set(OPENABLE_CLASS_LIST)

########################################################################################################################
# Actions
########################################################################################################################

# actions
IDX_TO_ACTION_TYPE = {
    0: "RotateLeft",
    1: "RotateRight",
    2: "MoveAhead",
    3: "LookUp",
    4: "LookDown",
    5: "OpenObject",
    6: "CloseObject",
    7: "PickupObject",
    8: "PutObject",
    9: "ToggleObjectOn",
    10: "ToggleObjectOff",
    11: "SliceObject",
}

ACTION_TYPE_TO_IDX = {v: k for k, v in IDX_TO_ACTION_TYPE.items()}
ACTION_TYPES = [IDX_TO_ACTION_TYPE[i] for i in range(len(IDX_TO_ACTION_TYPE))]

NAV_ACTION_TYPES = [
    "RotateLeft",
    "RotateRight",
    "MoveAhead",
    "LookUp",
    "LookDown"
]

INTERACT_ACTION_TYPES = [
    "OpenObject",
    "CloseObject",
    "PickupObject",
    "PutObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject"
]
