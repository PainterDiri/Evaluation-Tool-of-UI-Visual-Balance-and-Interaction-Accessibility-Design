import traceback
try:
    import cv2
    print("cv2_ok", cv2.__version__)
except Exception:
    traceback.print_exc()
