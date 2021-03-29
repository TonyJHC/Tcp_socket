import cv2
import numpy as np
#import tqdm
import os
import socket
from queue import Queue


class VideoStitcher:
    def __init__(self, video_out_path, video_out_width=1280, video_out_height=720, display=True):
        # Initialize arguments
        self.video_out_path = video_out_path
        self.video_out_width = video_out_width
        self.video_out_height = video_out_height
        self.display = display

        # Initialize the saved homography matrix
        self.saved_homo_matrix = None

    def stitch(self, images, ratio=0.75, reproj_thresh=4.0):
        # Unpack the images
        (image_b, image_a) = images

        # If the saved homography matrix is None, then we need to apply keypoint matching to construct it
        if self.saved_homo_matrix is None:
            # Detect keypoints and extract
            (keypoints_a, features_a) = self.detect_and_extract(image_a)
            (keypoints_b, features_b) = self.detect_and_extract(image_b)

            # Match features between the two images
            matched_keypoints = self.match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio,
                                                     reproj_thresh)

            # If the match is None, then there aren't enough matched keypoints to create a panorama
            if matched_keypoints is None:
                print('There are  no matches')
                return None

            # Save the homography matrix
            self.saved_homo_matrix = matched_keypoints[1]

        # Apply a perspective transform to stitch the images together using the saved homography matrix
        output_shape = (image_a.shape[1] + image_b.shape[1], image_a.shape[0])
        result = cv2.warpPerspective(image_a, self.saved_homo_matrix, output_shape)
        result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b

        # Return the stitched image
        return result

    @staticmethod
    def detect_and_extract(image):
        # Detect and extract features from the image (DoG keypoint detector and SIFT feature extractor)
        sift = cv2.xfeatures2d.SIFT_create()
        (keypoints, features) = sift.detectAndCompute(image, None)

        # Convert the keypoints from KeyPoint objects to numpy arrays
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        # Return a tuple of keypoints and features
        return (keypoints, features)

    @staticmethod
    def match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh):
        # Compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(features_a, features_b, k=2)
        matches = []

        for raw_match in raw_matches:
            # Ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(raw_match) == 2 and raw_match[0].distance < raw_match[1].distance * ratio:
                matches.append((raw_match[0].trainIdx, raw_match[0].queryIdx))

        # Computing a homography requires at least 4 matches
        if len(matches) > 4:
            # Construct the two sets of points
            points_a = np.float32([keypoints_a[i] for (_, i) in matches])
            points_b = np.float32([keypoints_b[i] for (i, _) in matches])

            # Compute the homography between the two sets of points
            (homography_matrix, status) = cv2.findHomography(points_a, points_b, cv2.RANSAC, reproj_thresh)

            # Return the matches, homography matrix and status of each matched point
            return (matches, homography_matrix, status)

        # No homography could be computed
        return None

    @staticmethod
    def draw_matches(image_a, image_b, keypoints_a, keypoints_b, matches, status):
        # Initialize the output visualization image

        (height_a, width_a) = image_a.shape[:2]
        (height_b, width_b) = image_b.shape[:2]
        visualisation = np.zeros((max(height_a, height_b), width_a + width_b, 3), dtype="uint8")
        visualisation[0:height_a, 0:width_a] = image_a
        visualisation[0:height_b, width_a:] = image_b

        for ((train_index, query_index), s) in zip(matches, status):
            # Only process the match if the keypoint was successfully matched
            if s == 1:
                # Draw the match
                point_a = (int(keypoints_a[query_index][0]), int(keypoints_a[query_index][1]))
                point_b = (int(keypoints_b[train_index][0]) + width_a, int(keypoints_b[train_index][1]))
                cv2.line(visualisation, point_a, point_b, (0, 255, 0), 1)

        # return the visualization
        return visualisation

    def get_bytes_stream(self, sock, length):
        # [참고]https://medium.com/@devfallingstar/python-python%EC%97%90%EC%84%9C-socket%EC%9C%BC%EB%A1%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EC%A3%BC%EA%B3%A0-%EB%B0%9B%EC%9D%84-%EB%95%8C-%EA%B0%92%EC%9D%84-%EB%81%9D%EA%B9%8C%EC%A7%80-recv%ED%95%98%EC%A7%80-%EB%AA%BB%ED%95%98%EB%8A%94-%EB%AC%B8%EC%A0%9C-ed1830a0a4a6
        buf = b''  # 바이트(인코딩 지정) 객체 생성 [참고]https://dojang.io/mod/page/view.php?id=2462
        while length:  # 지정한 bytes 길이까지 받기
            new_buf = sock.recv(length)  # recv(소켓에서 데이터 수신)는 데이터를 byte로 수신(네트워크 버퍼에서 작동) recv --> 0바이트 반환시 server측 socket과 통신이 안되고 있다는 증거

            if not new_buf:
                return None

            buf += new_buf
            length -= len(new_buf)

        return buf


    def run(self):
        # Set up video capture
        #left_video = cv2.VideoCapture(self.left_video_in_path)
        #right_video = cv2.VideoCapture(self.right_video_in_path)

        # Get information about the videos
        #n_frames = min(int(left_video.get(cv2.CAP_PROP_FRAME_COUNT)),
                      # int(right_video.get(cv2.CAP_PROP_FRAME_COUNT)))

        queue_left = Queue()
        queue_right = Queue()

        fps = 30
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        dim = (self.video_out_width, self.video_out_height)
        vid_writer = cv2.VideoWriter(self.video_out_path, codec, fps, dim)

        while True:

            message_left = 'please left frame'  # server socket에 보낼 메시지
            message_right = 'please right frame'

            left_client_socket.send(message_left.encode())  # server socket에 보내기(메시지 보내는 행위가 request)
            right_client_socket.send(message_right.encode())  # server socket에 보내기(메시지 보내는 행위가 request)

            left_length = self.get_bytes_stream(left_client_socket, 16)  # 16byte씩 바이트 길이를 수신 ex) length = b'57032 {bytes: 16}
            right_length = self.get_bytes_stream(right_client_socket,16)  # 16byte씩 바이트 길이를 수신 ex) length = b'57032 {bytes: 16}

            left_stringData = self.get_bytes_stream(left_client_socket, int(left_length))  # stringData = b'....' {bytes: 57032}
            right_stringData = self.get_bytes_stream(right_client_socket, int(right_length))  # stringData = b'....' {bytes: 57032}

            left_data = np.frombuffer(left_stringData, dtype='uint8')  # byte stream을 다시 1차원 array로
            right_data = np.frombuffer(right_stringData, dtype='uint8')

            left_decode_img = cv2.imdecode(left_data, cv2.IMREAD_COLOR)  # 1차원 array를 다시 img로 decode
            right_decode_img = cv2.imdecode(right_data, cv2.IMREAD_COLOR)  # 1차원 array를 다시 img로 decode

            #cv2.imshow('Image_client_left', left_decode_img)
            #cv2.imshow('Image_client_right', right_decode_img)
            #output_left.write(left_decode_img)
            #output_right.write(right_decode_img)

            #left : right = 1 : 1 stitching을 위한 queue
            queue_left.put(left_decode_img)
            queue_right.put(right_decode_img)

            # Grab the frames from their respective video streams
            #ok, left = left_video.read()
            # _, right = right_video.read()

            #큐에서 항목을 제거하고 반환. 큐가 비어 있으면, 항목이 들어올 때까지 기다림
            left = queue_left.get()
            right = queue_right.get()


             # Stitch the frames together to form the panorama
            stitched_frame = self.stitch([left, right])

            # No homography could not be computed
            if stitched_frame is None:
                print("[INFO]: Homography could not be computed!")
                break

                # Add frame to video
            stitched_frame = cv2.resize(stitched_frame, dim)

            if self.display:
                    # Show the output images
               cv2.imshow("Result", stitched_frame)

                # If the 'q' key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
               break

            vid_writer.write(stitched_frame)

        cv2.destroyAllWindows()
        print('[INFO]: Video stitching finished')

        # Save video
        print('[INFO]: Saving {} in {}'.format(self.video_out_path.split('/')[-1],
                                               os.path.dirname(self.video_out_path)))


if __name__=='__main__':
    # Example call to 'VideoStitcher'
    left_HOST = '192.168.1.25'  # localhost'  #라즈베리파이 IP
    left_PORT = 8888
    right_HOST = '192.168.1.26'  # localhost'  #라즈베리파이 IP
    right_PORT = 8090

    # output_left = cv2.VideoWriter('./result/socket_result_left.mp4',fourcc,fps,(640,480))
    # output_right = cv2.VideoWriter('./result/socket_result_right.mp4',fourcc,fps,(640,480))

    # 소켓 생성(패밀리, 소켓 타입)2
    # 패밀리 : AF_INET(IP4v) , AF_INET6(IP6v)
    # 소켓 타입 :  SOCK_STREAM(TCP)
    left_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    right_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 소켓 연결
    left_client_socket.connect((left_HOST, left_PORT))
    right_client_socket.connect((right_HOST, right_PORT))

    stitcher = VideoStitcher(video_out_path='./result/stitch_client.mp4')
    stitcher.run()
