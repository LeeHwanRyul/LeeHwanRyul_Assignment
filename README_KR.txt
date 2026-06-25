실행 순서

1. 이 폴더에서 터미널을 연다.
2. 필요한 패키지를 설치한다.
   pip install numpy matplotlib pillow jupyter
3. 학습한다.
   python train.py
4. 학습이 끝나면 Lee_mnist_model.pkl 파일이 생성된다.
5. 손글씨 사진 50장을 images 폴더에 넣는다.
   파일명은 0_1.png ~ 9_5.png 형식이다.
6. 손글씨 테스트를 실행한다.
   python module07.py
7. 제출 전 module07.ipynb도 처음부터 끝까지 실행하여 출력 결과를 저장한다.

주의
- 손글씨 사진은 과제 지시대로 직접 작성한 사진을 넣어야 한다.
- 인터넷 연결이 있어야 MNIST 데이터가 처음 한 번 다운로드된다.
- 다운로드가 막히면 mnist_data.py의 URL을 아래 주소로 바꿀 수 있다.
  https://raw.githubusercontent.com/fgnt/mnist/master/
