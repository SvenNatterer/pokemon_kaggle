import ctypes
import os
import platform
    
class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]

class SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int)
    ]

class V6ObservationBuffer(ctypes.Structure):
    _fields_ = [
        ("entity_ids", ctypes.c_int * 12),
        ("entity_features", ctypes.c_float * (12 * 36)),
        ("entity_tool_ids", ctypes.c_int * 12),
        ("entity_pre_evolution_ids", ctypes.c_int * (12 * 3)),
        ("entity_energy_card_ids", ctypes.c_int * (12 * 8)),
        ("hand_ids", ctypes.c_int * 24),
        ("discard_ids", ctypes.c_int * (2 * 30)),
        ("revealed_ids", ctypes.c_int * 120),
        ("prize_ids", ctypes.c_int * (2 * 6)),
        ("search_ids", ctypes.c_int * 60),
        ("looking_ids", ctypes.c_int * 60),
        ("own_deck_ids", ctypes.c_int * 60),
        ("context_card_ids", ctypes.c_int * 3),
        ("log_card_ids", ctypes.c_int * 10),
        ("option_card_ids", ctypes.c_int * 65),
        ("option_attack_ids", ctypes.c_int * 65),
        ("option_types", ctypes.c_int * 65),
        ("option_areas", ctypes.c_int * 65),
        ("option_features", ctypes.c_float * (65 * 21)),
        ("vector", ctypes.c_float * 1500),
        ("aux_target", ctypes.c_float * 2000),
        ("action_mask", ctypes.c_int8 * 66),
    ]

os_name = platform.system()
if os_name == 'Windows':
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cg.dll")
elif os_name == "Darwin":
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcg.dylib")
elif platform.machine() in ('arm64', 'aarch64'):
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcg-arm64.so")
else:
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcg.so")
lib = ctypes.cdll.LoadLibrary(lib_path)

lib.GameInitialize()

lib.BattleStart.restype = StartData
lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]

lib.AgentStart.restype = ctypes.c_void_p

lib.BattleFinish.argtypes = [ctypes.c_void_p]

lib.GetBattleData.restype = SerialData
lib.GetBattleData.argtypes = [ctypes.c_void_p]

lib.GetV6Observation.restype = ctypes.c_int
lib.GetV6Observation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(V6ObservationBuffer)]

lib.Select.restype = ctypes.c_int
lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.VisualizeData.restype = ctypes.c_char_p
lib.VisualizeData.argtypes = [ctypes.c_void_p]

lib.SearchBegin.restype = ctypes.c_char_p
lib.SearchBegin.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int]

lib.SearchStep.restype = ctypes.c_char_p
lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.SearchEnd.argtypes = [ctypes.c_void_p]

lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]

lib.AllCard.restype = ctypes.c_char_p

lib.AllAttack.restype = ctypes.c_char_p

class Battle:
    battle_ptr = None
    obs = None
