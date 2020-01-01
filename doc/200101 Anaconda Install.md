# Anaconda Install

### 1. 아나콘다 설치
https://www.anaconda.com/distribution 에서 설치파일 다운로드
호환을 위해 windows 32bit 버전으로 설치

Add Anaconda to the system PATH environment variable
Register Anaconda as the system Python 3.7
둘 다 체크

cmd 화면에서 오류남
-> 이전에 파이썬 설치 시 남아있는 환경변수와 충돌 때문. 해당 환경변수를 삭제하거나 conda에 맞춰 경로 수정

### 2. 가상환경 설치

```bash
#가상환경 생성
conda create -n 환경명 python=버전

conda env list #환경을 제대로 만들었는지 확인
conda info --envs #확인

activate 환경명
#만약 위가 안된다면 아래와 같이 하자
source activate 환경명

#환경 종료
conda deactivate

#가상환경 삭제
conda remove --name 환경명 --all
#혹은
conda remove -n 가상환경명 --all
```

### 3. Windows에서 크롬으로 Jupyter 노트북을 여는 방법

* 주피터 노트북 설치는 `conda install jupyter ` 이나, 아나콘다에는 이미 설치되어 있다.

* 아나콘다 프롬프트에서 `jupyter notebook --generate-config`을 실행하여 노트북 설정 파일을 생성 홈 디렉토리를 얻는다.

나는 user folder/.jupyter/ 안에 있었다.

해당 디렉토리로 가서 `.jupyter` 폴더에있는 파일 `jupyter_notebook_config.py`을 편집하자.

크롬 브라우저 주소를 찾고, \는 /로 바꾸자

`#c.NotebookApp.Browser`를 찾아서 다음과 같이 변경하고 저장한다.

```
c.NotebookApp.browser = 'c:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
```



##  라이브러리 패키지

### 1. 기본 라이브러리 업데이트
처음 anaconda를 설치하면 높은 확률로 라이브러리 업데이트가 반영되지 않은 버전이 설치되었을 것이다.

``` bash
conda update conda
conda update anaconda
conda update python
```

혹은 아래 한 줄로도 가능

```bash
conda update --all
```

### 2. 새로운 라이브러리 설치

pip와 비슷하게. 새로운 라이브러리 설치는

```bash
conda install 이름
```

### 3. 라이브러리 업데이트

아나콘다 자체 업데이트를 먼저 해야한다.

그렇지 않으면, anaconda를 지우고 새 버전으로 다시 까는 귀찮음을 감수해야 한다.

```bash
conda update -n root conda
```

그 후 패키지 업데이트

```bash
conda update --all
```

### 4. 라이브러리 삭제

```bash
conda remove 이름
```

### 5. pip란

#### pip v.s. pip3

`pip` : anaconda3에서 관리하고 있는 전역 pip에 들어가게 된다.

`pip2` : local 내에 깔린 pip2 (python2 버전)에 들어가게 된다.

`pip3` : local 내에 깔린 pip3 (python3 버전)에 들어가게 된다.

####  conda 내에서의 pip install

```bash
#가상환경 만든 후 들어감
$ conda create -n py36 python=36
...
$ source activate py36
$ (py36) ~
```

conda 가상환경 안에서 pip 명령어의 차이를 보면...

`pip` : py36 환경의 pip에 들어가게 된다.

`pip2` : local 내에 깔린 pip2 (python2 버전)에 들어가게 된다.

`pip3` : local 내에 깔린 pip3 (python3 버전)에 들어가게 된다.

즉, 가상환경 내에서만 패키지를 설치하려면,  `pip install ~`만 해야한다.

만약 `pip3 install`로 하면, 전역 local에 깔리는 것이다.

#### conda install v.s. pip install

conda install은 현재 가상환경에만 패키지를 설치한다. pip도 동일하다.

즉, 일반적으로 원하는 패키지를 현재 환경에만 설치하기 위해 `conda install`이나 `pip install`이나 같다.

