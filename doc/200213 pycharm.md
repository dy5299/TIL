# PyCharm 파이참에서 아나콘다, 가상환경 설정하기

PyCharm에서 프로젝트별 작업환경을 구분하여 작업하기 위해 미리 기본 설정이 조금 필요합니다.

Anaconda는 미리 설치되어있다고 가정하고 파이참에서 Virtual Enviroment를 연동하는 방법을 알아보겠습니다.

## Setting - 아나콘다 환경으로 설정

파이참에서 Configure -> Settings


![200213_pycharm1](images/200213_pycharm1.PNG)

왼쪽에 설정할 프로젝트명 아래에 Project Interpreter로 들어갑니다.

오른쪽에 Project Interpreter 드롭메뉴에 설정할 가상환경이 없으면 오른쪽 톱니바퀴(설정) 버튼을 누릅니다.







뜬 팝업창에서 Project Interpreter -> Conda Environment -> New environment or Existing environment 클릭

![200213_pycharm2](images/200213_pycharm2.PNG)

가상환경을 여기서 새로 만들고 싶으면 New environment에서 파이썬 버전을 설정하시면 되고

저는 사용할 가상환경이 미리 만들어져 있어서 Existing environment에서 interpreter에서 설정하였습니다.

경로는 파이썬 기본설정으로 설치했으면 아래 경로와 같을 것입니다.

`C:\Users\<사용자명>\Anaconda3\envs\가상환경이름`

Make available to all projects는 필요하신 분들은 체크하세요. 저는 파이썬 3.6버전 전용 가상환경이라 적용했습니다.



## Setting - Terminal에서 Virtual ENV 쓰도록 설정하기

파이참 안에서도 터미널을 사용할 수 있습니다.

그런데 위에서 설정한 프로젝트 가상환경 설정이 알아서 자동으로 적용되지 않습니다.

파이참 안에서 해당 환경으로 터미널을 사용하시려면 따로 설정해주어야 합니다.



세팅창에서 Tools > Terminal

![200213_pycharm3](images/200213_pycharm3.PNG)

Shell path 부분을 activate.bat 경로를 찾아 다음과 같이 수정해줍니다.

activate.bat 경로는 OS마다 다를 수 있으므로 직접 검색해서 작성해주세요.

`cmd.exe "/K" C:\ProgramData\Anaconda3\Scripts\activate.bat 가상환경이름`



Terminal에서 기본으로 왼쪽에 `(가상환경이름)` 으로 뜨면 잘 설정된 것입니다.

