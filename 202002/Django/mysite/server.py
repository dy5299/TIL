import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#소켓이 꼭 TCPIP만 의미하는 것은 아니고 여러가지 통신방법이 있다.
#파라미터가 두 개 - 첫번째는 IP를 쓰겠다
#두번째는 TCP/UCP인데 STREAM은 TCP방법을 쓰겠다

server_socket.bind(('localhost', 80))    #IP, 포트번호
server_socket.listen(0)                     #포트번호를 listening. 동시에 연결할 최대 소켓 갯수. 0은 automatically
#대기모드
print('listening...')

client_socket, addr = server_socket.accept() #클라이언트 접속될 때까지는 대기상태
print('accepting')
data =client_socket.recv(65535) #클라이언트 접속이 되면 데이터를 읽어들임.
#데이터는(패킷은) 최대 64k. 더 작을수도 클 수도 있는데, 크면 쪼개서 전송된다.

print('receive >> ' + data.decode())  #unicode to 한글

client_socket.send(data)
print('send data')
client_socket.close()
print('종료')