
import cv2
import imutils
import argparse

HOGCV = cv2.HOGDescriptor() # SVM classifier
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # calls the pre-trained model for human detection

def detect(frame):
    (bounding_box_coords, _) = HOGCV.detectMultiScale(frame, winStride = (4,4), padding = (8,8), scale = 1.03)
    # detectMultiScale detects objects of different sizes
    # threshold values, padding, and window stride numbers can be changed
    
    person = 1
    for x, y, w, h in bounding_box_coords:  # coordinates of the bounding box for a person
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2) # draws the frame
        cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1
    
    # counter and text you'll see when running the program with an image
    cv2.putText(frame, 'Status: Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons: {person-1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)
    
    return frame

def humanDetector(args):    # sets images, videos to their respective arguments/paths
    img_path = args["image"]
    vid_path = args['video']
    output_path = args['output']
    if str(args["camera"]) == 'True':   # depends if the default state is set to True/False (check the argsParser function)
        camera = True
    else:
        camera = False
    
    writer = None
    if args['output'] is not None and img_path is None:
        writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))   # saves the video if it can
    
    if camera:
        detectByCamera(output_path, writer)
    elif vid_path is not None:
        detectByPathVideo(vid_path, writer)
    elif img_path is not None:
        detectByPathImage(img_path, args['output'])

def detectByCamera(writer):
    video = cv2.VideoCapture(0) # passing in 0 means we want to record from a webcam
    print('Looking for humans...')
    
    while True:
        check, frame = video.read() # reads frame by frame
        
        frame = detect(frame)
        if writer is not None:
            writer.write(frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

def detectByPathVideo(path, writer):    # you can use a video for the program in cmd by placing -v before your video path
    video = cv2.VideoCapture(path)  # make sure your video is readable (.mp4)
    check, frame = video.read()
    if check == False:
        print('Video not found. A full path for your video must be provided.')
        return
    
    print('Looking for humans...')
    while video.isOpened():
        check, frame = video.read()
        
        if check:   # will try to draw boxes for the people it finds in the video (usually frame by frame)
            frame = imutils.resize(frame, width = min(800, frame.shape[1]))
            frame = detect(frame)
        
            if writer is not None:
                writer.write(frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    
    video.release()
    cv2.destroyAllWindows()

def detectByPathImage(path, output_path):   # you can use an image for the program in cmd by placing -i before your image path
    img = cv2.imread(path)
    
    img = imutils.resize(img, width = min(800, img.shape[1]))
    
    result_img = detect(img)    # attempts to find people in said image
    
    if output_path is not None:
        cv2.imwrite(output_path, result_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def argsParser():   # dictionary of passed arguments. these arguments will be used in cmd in order to test out images and videos
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default = None, help = "path to Video File ")
    arg_parse.add_argument("-i", "--image", default = None, help = "path to Image File ")
    arg_parse.add_argument("-c", "--camera", default = True, help = "Set true if you want to use a camera. ")   # no clue if the camera works
    arg_parse.add_argument("-o", "--output", type = str, help = "path to optional output video file ")  # type -o '___.jpg' in cmd if you want to save your output image
    args = vars(arg_parse.parse_args())
    
    return args

if __name__ == '__main__':  # tells python to run the code
    HOGCV = cv2.HOGDescriptor() # declared here in the main function
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    args = argsParser()
    humanDetector(args)

