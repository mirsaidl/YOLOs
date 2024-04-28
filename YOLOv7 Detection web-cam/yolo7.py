from ikomia.dataprocess.workflow import Workflow
from ikomia.utils import ik
from ikomia.utils.displayIO import display
import cv2

stream = cv2.VideoCapture(0)

# Init the workflow
wf = Workflow()

# Add color conversion
cvt = wf.add_task(ik.ocv_color_conversion(code=str(cv2.COLOR_BGR2RGB)), auto_connect=True)

# Add YOLOv7 detection
yolo = wf.add_task(ik.infer_yolo_v7(conf_thres="0.7"), auto_connect=True)

while True:
    ret, frame = stream.read()
    
    # Test if streaming is OK
    if not ret:
        continue

    # Run workflow on image
    wf.run_on(frame)

    # Display results from "yolo"
    display(
        yolo.get_image_with_graphics(),
        title="Object Detection - press 'q' to quit",
        viewer="opencv"
    )

    # Press 'q' to quit the streaming process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the stream object
stream.release()

# Destroy all windows
cv2.destroyAllWindows()