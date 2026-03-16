from resram_core import load_input

try:
    obj = load_input()
    print("load_input() successful")
    print(f"E0: {obj.E0}")
    print(f"gamma: {obj.gamma}")
    print(f"theta: {obj.theta}")
    print(f"M: {obj.M}")
    print(f"boltz_toggle: {obj.boltz_toggle}")
except Exception as e:
    print(f"Error in load_input(): {e}")
    import traceback
    traceback.print_exc()
