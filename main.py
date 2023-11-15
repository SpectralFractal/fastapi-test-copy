from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
from base64 import b64encode
import aiofiles
import os
from starlette.websockets import WebSocketDisconnect
import time
from multiprocessing import Pool
import torch
from starlette.middleware.cors import CORSMiddleware


# Global objects for better performance
detector = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./Haarcascades/haarcascade_eye.xml')


app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_headers=['*'],
        allow_methods=['*'],
        allow_origins=['*'],
    )

# Ensure the directory for saved frames exists
os.makedirs('/saved_frames', exist_ok=True)



@app.get("/")
def read_root():
    return {"Hello": "World"}

# using yolo model
@app.websocket("/ws-fast-yolo")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    processing_times = []
    frame_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()

            start_time = time.time()  # Start time of processing

            # Convert to a numpy array
            nparr = np.frombuffer(data, np.uint8)

            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            frame = torch.from_numpy(frame).unsqueeze(0)  # Add a batch dimension
            frame = frame.permute(0, 3, 1, 2)  # Rearrange the dimensions to CHW format

            if frame is not None:
                # Perform detection
                results = detect(frame.float() / 255.0)  # Normalize the frame to 0-1 range
                # Process the results to draw bounding boxes, etc.

                # Convert the tensor to a numpy array and encode it as a JPEG
                frame = results.render()[0]  # Render the detections
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR format expected by OpenCV
                success, buffer = cv2.imencode('.jpg', frame)
                frame_encoded = b64encode(buffer).decode('utf-8')
                processing_time = time.time() - start_time  # End time of processing
                processing_times.append(processing_time)
                frame_counter += 1
                print(f"Processing time: {processing_time} seconds")
                await websocket.send_text('data:image/jpeg;base64,' + frame_encoded)
            else:
                print('Frame is None')
    except WebSocketDisconnect:
        get_time(processing_times, frame_counter)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


# Endpoint for also saving files to disk on live
@app.websocket("/save-file")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0
    try:
        while True:
            # Wait for any message from the client
            data = await websocket.receive_bytes()
            # Increment frame counter
            frame_count += 1
            # Define file path
            file_path = f"saved_frames/frame_{frame_count}.jpg"
            # Save the frame asynchronously
            async with aiofiles.open(file_path, 'wb') as out_file:
                await out_file.write(data)
                print(f"Frame saved: {file_path}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


# best performance for sending processed images
@app.websocket("/ws-fast")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    processing_times = []
    frame_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()

            start_time = time.time()  # Start time of processing

            # Convert to a numpy array
            nparr = np.frombuffer(data, np.uint8)

            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                # Perform image processing as before
                faces = detector.detectMultiScale(frame, 1.3, 5)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 3)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                _, buffer = cv2.imencode('.jpg', frame)
                frame_encoded = b64encode(buffer).decode('utf-8')
                processing_time = time.time() - start_time  # End time of processing
                processing_times.append(processing_time)
                frame_counter += 1
                print(f"Processing time: {processing_time} seconds")


                # aici le trimite inapoi
                await websocket.send_text('data:image/jpeg;base64,' + frame_encoded)
            else:
                print('Frame is None')
    except WebSocketDisconnect:
        get_time(processing_times, frame_counter)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


# Endpoint for showing in real time the processed frames
@app.websocket("/ws-show")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            start_time = time.time()  # Start time of processing

            # Convert to a numpy array
            nparr = np.frombuffer(data, np.uint8)

            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                # Perform image processing as before
                detector = cv2.CascadeClassifier('/Haarcascades/haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier('/Haarcascades/haarcascade_eye.xml')

                # - This function detects faces in the frame. The parameters include the image, a scale factor (how
                # much the image size is reduced at each image scale), and minNeighbors (how many neighbors each
                # candidate rectangle should have to retain it). These parameters are used to adjust the detection
                # algorithm's sensitivity.
                # scaled_frame = cv2.resize(frame, dsize=(0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)

                faces = detector.detectMultiScale(frame, 1.3, 5)

                #  - This converts the frame to a grayscale image. Haar cascades require grayscale images to function.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    # eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                    # for (ex, ey, ew, eh) in eyes:
                    #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                processing_time = time.time() - start_time  # End time of processing
                print(f"Processing time: {processing_time} seconds")

                # Display the resulting frame
                cv2.imshow('Video', frame)

                # Press 'q' to close the video window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print('Frame is None')
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
        cv2.destroyAllWindows()


# Function used for processing image
def process_image(data):
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is not None:
        # Perform image processing as before
        detector = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('./Haarcascades/haarcascade_eye.xml')
        faces = detector.detectMultiScale(frame, 1.3, 5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 3)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    _, buffer = cv2.imencode('.jpg', frame)
    frame_encoded = b64encode(buffer).decode('utf-8')
    return 'data:image/jpeg;base64,' + frame_encoded


# endpoint used for testing performance with multithreading 1
@app.websocket("/ws-test1")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pool = Pool(processes=4)  # Pool of worker processes
    processing_times = []
    frame_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            start_time = time.time()

            # Offload the image processing to a pool of workers
            result = pool.apply_async(process_image, (data,))
            frame_encoded = result.get()  # This will block until the result is ready

            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            frame_counter += 1
            print(f"Processing time: {processing_time} seconds")
            await websocket.send_text(frame_encoded)

    except WebSocketDisconnect:
        get_time(processing_times, frame_counter)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        pool.close()
        pool.join()
        await websocket.close()


# endpoint used for testing performance with multithreading 2
@app.websocket("/ws-test2")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pool = Pool(processes=12)  # Pool of worker processes
    pending_results = []
    processing_times = []
    frame_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            start_time = time.time()

            async_result = pool.apply_async(process_image, (data,))
            pending_results.append((start_time, async_result))

            for start_time, result in pending_results[:]:
                if result.ready():
                    frame_encoded = result.get()
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    frame_counter += 1
                    print(f"Processing time: {processing_time} seconds")
                    await websocket.send_text(frame_encoded)
                    pending_results.remove((start_time, result))

    except WebSocketDisconnect:
        get_time(processing_times, frame_counter)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        get_time(processing_times, frame_counter)

        pool.close()
        pool.join()
        await websocket.close()


# Function used for getting the average time for a frame render
def get_time(processing_times=None, frame_counter=None):
    total_processing_time = sum(processing_times)
    print(f"Total processing time for all images: {total_processing_time} seconds")
    print(f"Total frames processed: {frame_counter}")

    if frame_counter > 0:
        average_processing_time = total_processing_time / frame_counter
        print(f"Average processing time per frame: {average_processing_time} seconds")
    else:
        print("No frames were processed.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5000)
