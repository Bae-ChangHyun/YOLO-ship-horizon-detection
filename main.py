import os
import cv2
import numpy as np
from ultralytics import YOLO

# size of the display frame
display_xSize = 960
display_ySize = 540

kernelSize = (1,1) # 에로션연산에 사용되는 커널의 크기입니다. 커널의 크기가 클수록 더 많은 픽셀이 제거됩니다.
kSize = (5,5) # 가우시안 블러링에 사용되는 커널의 크기입니다. 커널의 크기가 클수록 블러링 효과가 강해집니다.
sigmaX = 4 # 가우시안 블러링에서 X 방향의 표준 편차입니다. 값이 클수록 블러링 효과가 강해집니다.
CannyThreshold1 = 100 # 캐니 엣지 검출에서 첫 번째 임계값입니다. 낮을수록 더 많은 에지가 검출됩니다.
CannyThreshold2 = 150 # 캐니 엣지 검출에서 두 번째 임계값입니다. 높을수록 더 적은 에지가 검출됩니다.
CannyApertureSize = 5 # 캐니 엣지 검출에서 사용되는 소벨 연산자의 커널 크기입니다. 홀수여야 합니다
ErosionIterations = 4 # 에로션 연산의 반복 횟수입니다. 반복 횟수가 많을수록 더 많은 픽셀이 제거됩니다
HoughThreshold = 51 # 허프 변환에서 선을 검출하기 위한 최소 투표 수입니다. 값이 높을수록 더 강한 선만 검출됩니다.
HoughMinLineLength = 0 # 허프 변환에서 검출할 선의 최소 길이입니다. 값이 클수록 더 긴 선만 검출됩니다
HoughMaxLineGap = 0 # 허프 변환에서 선을 연결할 최대 간격입니다. 값이 클수록 더 멀리 떨어진 선도 연결됩니다.

def setkernelSize(size):
    global kernelSize
    kernelSize = (size, size)

def setkSize(size):
    global kSize
    kSize = (5,5)

def setsigmaX(sigma):
    global sigmaX
    sigmaX = sigma

def setCannyThreshold1(threshold):
    global CannyThreshold1
    CannyThreshold1 = threshold

def setCannyThreshold2(threshold):
    global CannyThreshold2
    CannyThreshold2 = threshold

def setCannyApertureSize(size):
    global CannyApertureSize
    if size < 3:
        size = 3
    elif size % 2 == 0:
        size -= 1
    CannyApertureSize = size
    cv2.setTrackbarPos('Canny Aperture Size', 'Display frame', size)

def setErosionIteration(iters):
    global ErosionIterations
    ErosionIterations = iters

def setHoughThreshold(threshold):
    global HoughThreshold
    HoughThreshold = threshold

def setHoughMinLineLength(length):
    global HoughMinLineLength
    HoughMinLineLength = length

def setHoughMaxLineGap(gap):
    global HoughMaxLineGap
    HoughMaxLineGap = gap

def init_control():
    cv2.namedWindow('Display frame')
    cv2.createTrackbar('erosion Kernel Size', 'Display frame', 1, 20, setkernelSize)
    cv2.createTrackbar('blurring k Size', 'Display frame', 5, 20, setkSize)
    cv2.createTrackbar('blurring sigmaX', 'Display frame', sigmaX, 20, setsigmaX)
    cv2.createTrackbar('Canny Threshold1', 'Display frame', CannyThreshold1, 250, setCannyThreshold1)
    cv2.createTrackbar('Canny Threshold2', 'Display frame', CannyThreshold2, 250, setCannyThreshold2)
    cv2.createTrackbar('Canny Aperture Size', 'Display frame', CannyApertureSize, 7, setCannyApertureSize)
    cv2.createTrackbar('Erosion Iterations', 'Display frame', ErosionIterations, 20, setErosionIteration)
    cv2.createTrackbar('Hough Threshold', 'Display frame', HoughThreshold, 255, setHoughThreshold)
    cv2.createTrackbar('Hough Min Line Length', 'Display frame', HoughMinLineLength, 255, setHoughMinLineLength)
    cv2.createTrackbar('Hough Max Line Gap', 'Display frame', HoughMaxLineGap, 255, setHoughMaxLineGap)

def draw_boxes(frame, boxes, line, margin):
    """
    Draws boxes on the given frame.

    Args:
        frame (numpy.ndarray): The input frame.
        boxes (list): List of boxes to draw. Each box is represented as [x1, y1, x2, y2].
        line (tuple): Coordinates of the horizon line in the format (x1, y1, x2, y2).
        margin (int): Margin value for the horizon line.

    Returns:
        numpy.ndarray: The frame with boxes drawn on it.
    """
    global debug
    if line is not None: 
        x1, y1, x2, y2 = line
        margin_vector = np.array([-(y2 - y1), x2 - x1])
        margin_vector = margin_vector / np.linalg.norm(margin_vector) * margin

        line_top = (int(x1 - margin_vector[0]), int(y1 - margin_vector[1]), int(x2 - margin_vector[0]), int(y2 - margin_vector[1]))
        line_bottom = (int(x1 + margin_vector[0]), int(y1 + margin_vector[1]), int(x2 + margin_vector[0]), int(y2 + margin_vector[1]))

        if debug: 
            cv2.line(frame, (line_top[0], line_top[1]), (line_top[2], line_top[3]), (0, 165, 255), 2)         # lines upper range
            cv2.line(frame, (line_bottom[0], line_bottom[1]), (line_bottom[2], line_bottom[3]), (0, 165, 255), 2)  # lines lower range

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            if y1 >= line_top[1] and y2 <= line_bottom[1]:
                color = (255, 0, 0)                                                             # if box is in the lines range, shows blue
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            else:
                color = (0, 0, 255)  # if box is not in the lines range, shows red(shows when debug is True)
                if debug:cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    else:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
    return frame

def inference_run(input_path, output_path, model_id, conf_threshold=0.5, iou_threshhold=0.7, margin=40, frame_limit=None):
    """
    Runs inference on the input video and saves the output video with detected boxes.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.(save only if save is True)
        model_id (str): Identifier for the YOLO model to use.
        conf_threshold (float, optional): Confidence threshold for detecting objects. Defaults to 0.5.
        iou_threshhold (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.7.
        margin (float, optional): Margin value for the horizon line. Defaults to 0.4.
        frame_limit (int, optional): Limit the number of frames to process for test. Defaults to None.(max = total_frames)

    Returns:
        str: Path to the saved output video file.
    """
    global debug, save
    processed_frames = 0
    if debug: init_control()
    model = YOLO(model_id)
    lines_list = []

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        
        if frame_limit:
            frame_limit = min(frame_limit, total_frames)
            if processed_frames > frame_limit: break
            
        ret, frame = cap.read() # input_video frame
        if not ret: break
        boxes = model.predict(source=frame, conf=conf_threshold, iou=iou_threshhold, device='cuda:0')[0].boxes.xyxy.tolist()
        
        display_frame = cv2.resize(frame, (display_xSize, display_ySize), interpolation=cv2.INTER_AREA) # frame for controls
        
        frame_gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        
        erosion = cv2.erode(frame_gray, kernel=np.ones(kernelSize, np.uint8), iterations=ErosionIterations)
        
        blurred = cv2.GaussianBlur(erosion, ksize=kSize, sigmaX=sigmaX)

        edges = cv2.Canny(blurred, threshold1=CannyThreshold1, threshold2=CannyThreshold2, apertureSize=CannyApertureSize)
        
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=HoughThreshold, minLineLength=HoughMinLineLength, maxLineGap=HoughMaxLineGap)
        lines_frame = np.zeros_like(display_frame)  # frame for only lines
        
        if lines is not None:
            print("Lines detected")
            x1_max, x2_max, y1_max, y2_max = 0, 0, 0, 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (x2 - x1) ** 2 + (y2 - y1) ** 2 > (x2_max - x1_max) ** 2 + (y2_max - y1_max) ** 2:
                    y1_max = y1
                    y2_max = y2
                    x1_max = x1
                    x2_max = x2
                lines_list.append((x1, y1, x2, y2))
                if len(lines_list) > 10:
                    lines_list.pop(0)
                cv2.line(lines_frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
        else:
            if len(lines_list) > 0:
                avg_line = np.mean(lines_list, axis=0).astype(int)
                x1, y1, x2, y2 = avg_line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
        slope = -(y2_max - y1_max) / (x2_max - x1_max)

        y1_max = y1_max - slope * -x1_max
        y1_max = int(y1_max)
        
        y2_max = y2_max - slope * (display_xSize - x2_max)
        y2_max = int(y2_max)

        x1_max = 0
        x2_max = display_xSize
        
        cv2.line(display_frame, pt1=(x1_max, y1_max), pt2=(x2_max, y2_max), color=(0, 0, 255), thickness=2)
        
        boxes = model.predict(source=frame, conf=conf_threshold, iou=iou_threshhold, device='cuda:0')[0].boxes.xyxy.tolist()
        if debug:
            cv2.line(frame, pt1=(int(x1_max * frame_width / display_xSize), int(y1_max * frame_height / display_ySize)), 
                            pt2=(int(x2_max * frame_width / display_xSize), int(y2_max * frame_height / display_ySize)), 
                            color=(0, 0, 255), thickness=2)
        
        # convert boxes to display frames size
        display_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1 * display_xSize / frame_width)
            y1 = int(y1 * display_ySize / frame_height)
            x2 = int(x2 * display_xSize / frame_width)
            y2 = int(y2 * display_ySize / frame_height)
            display_boxes.append([x1, y1, x2, y2])
        
        # draw boxes on the main frame
        frame = draw_boxes(frame, boxes, (int(x1_max * frame_width / display_xSize), int(y1_max * frame_height / display_ySize), int(x2_max * frame_width / display_xSize), int(y2_max * frame_height / display_ySize)), margin)
        
        # draw boxes on the display frame
        if debug: 
            display_frame = draw_boxes(display_frame, display_boxes, (x1_max, y1_max, x2_max, y2_max), margin)
            cv2.imshow('Display frame', display_frame)
            cv2.imshow('lines', lines_frame)
            #cv2.imshow('Origin', frame)
        
        if cv2.waitKey(10) == 27: break

        if save:out.write(frame)

        processed_frames += 1
        print(f"Total frames: {total_frames}, Processed frames: {processed_frames}")

    cap.release()
    if save: out.release()
    cv2.destroyAllWindows()

    return output_path

if __name__ == "__main__":
    debug = True # if True, show the debug window and draw line on output
    save = False # Save the output video

    input_path  = "video_input/test.mp4"
    output_path = "video_output/tmp.mp4" # save only if save is True
    model_path = "yolov10n" # yolo model name or model path
    
    conf_threshold = 0.35 # yolo confidence threshold
    iou_threshhold = 0.35 # yolo iou threshold
    margin = 40           # margin value for the horizon line 
    frame_limit = None    # Limit the number of frames to process for test
    result_path = inference_run(input_path, output_path, model_path, conf_threshold, iou_threshhold, margin, frame_limit)