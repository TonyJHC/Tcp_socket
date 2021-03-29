import socket
import numpy as np
import cv2
"""
1. 소켓 생성 socket()
2. 소켓 연결 connect()
"""

def get_bytes_stream(sock, length):
    #[참고]https://medium.com/@devfallingstar/python-python%EC%97%90%EC%84%9C-socket%EC%9C%BC%EB%A1%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EC%A3%BC%EA%B3%A0-%EB%B0%9B%EC%9D%84-%EB%95%8C-%EA%B0%92%EC%9D%84-%EB%81%9D%EA%B9%8C%EC%A7%80-recv%ED%95%98%EC%A7%80-%EB%AA%BB%ED%95%98%EB%8A%94-%EB%AC%B8%EC%A0%9C-ed1830a0a4a6
    buf = b'' #바이트(인코딩 지정) 객체 생성 [참고]https://dojang.io/mod/page/view.php?id=2462
    while length: #지정한 bytes 길이까지 받기
        new_buf = sock.recv(length) #recv(소켓에서 데이터 수신)는 데이터를 byte로 수신(네트워크 버퍼에서 작동) recv --> 0바이트 반환시 server측 socket과 통신이 안되고 있다는 증거

        if not new_buf:
            return None

        buf += new_buf
        length -= len(new_buf)

    return buf


HOST = '192.168.1.25' #localhost'  #라즈베리파이 IP
PORT = 8888

fps =30
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
output = cv2.VideoWriter('./result/socket_result_test.mp4',fourcc,fps,(640,480))

#소켓 생성(패밀리, 소켓 타입)
#패밀리 : AF_INET(IP4v) , AF_INET6(IP6v)
#소켓 타입 :  SOCK_STREAM(TCP)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#소켓 연결
client_socket.connect((HOST, PORT))
#i = 0

while True:

    message = 'process 1' # server socket에 보낼 메시지
    client_socket.send(message.encode()) # server socket에 보내기(메시지 보내는 행위가 request)

    length = get_bytes_stream(client_socket, 16) #16byte씩 바이트 길이를 수신 ex) length = b'57032 {bytes: 16}
    stringData = get_bytes_stream(client_socket, int(length)) #stringData = b'....' {bytes: 57032}
    data = np.frombuffer(stringData, dtype='uint8') #byte stream을 다시 1차원 array로

    decode_img = cv2.imdecode(data, cv2.IMREAD_COLOR) #1차원 array를 다시 img로 decode
    cv2.imshow('Image_client_right', decode_img)
    output.write(decode_img)
    #cv2.imwrite('./data/test/image' + str(i) + '.jpg',decode_img)
    #i = i + 1

    if cv2.waitKey(1)== 27:
        break

output.release()
cv2.destroyAllWindows()

client_socket.close()
