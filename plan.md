현재 코덱 코드베이스를 실제로 분석한 뒤, 기존 camera projection / depth 기반 GlobalMotion tool에 다음 구조를 구현하기 위한 구체적인 implementation plan을 작성해줘.

아직 코드를 수정하지 말고, 먼저 관련 파일·클래스·함수·syntax 흐름을 실제 코드에서 찾아서 계획만 세워라.

추측으로 파일명이나 함수명을 만들지 말고, 반드시 현재 코드에 존재하는 구조와 호출 흐름을 기준으로 작성해라.

모든 신규 코드, syntax, 상태 변수, debug 기능은 반드시 다음 macro 안에 있어야 한다.

#if GlobalMotion
...
#endif

1. 구현 목표

Camera intrinsic K, picture별 camera pose Rt, picture별 camera projection tool on/off 정보를 picture 또는 slice header syntax로 전달한다.

Depth와 camera geometry state는 codec GOP size나 IntraPeriod와 독립적으로 동작한다.

Depth 및 camera geometry reset period는 다음과 같이 32 picture로 고정한다.

constexpr int CAM_GEOMETRY_RESET_PERIOD = 32;

RA와 LDB 구조를 모두 지원해야 한다.

지원할 주요 설정은 다음과 같다.

RA / LDB
GOP size 변경 가능
IntraPeriod 변경 가능
FrameSkip 지원

FrameSkip은 아래 전제를 따른다.

FrameSkip % 32 == 0

FrameSkip 적용 후 실제 encoding이 시작되는 첫 picture의 codec POC는 항상 0이다.

예:

FrameSkip = 0   → source frame 0이 codec POC 0
FrameSkip = 32  → source frame 32가 codec POC 0
FrameSkip = 64  → source frame 64가 codec POC 0

따라서 decoder는 FrameSkip 값이나 source absolute frame index를 알 필요가 없다.

Decoder의 모든 geometry group과 reset 판단은 codec POC 기준으로 수행한다.

2. Geometry group 정의

Codec POC 기준으로 geometry group을 정의한다.

groupIdx  = poc / CAM_GEOMETRY_RESET_PERIOD;
anchorPoc = groupIdx * CAM_GEOMETRY_RESET_PERIOD;

예:

group 0: POC 0  ~ 32
group 1: POC 32 ~ 64
group 2: POC 64 ~ 96

경계 picture는 이전 group의 endpoint이면서 다음 group의 anchor가 될 수 있다.

예를 들어 POC 32는 다음 두 geometry 표현을 가질 수 있다.

이전 group 기준:
  K = K0
  Rt = Rt(0 → 32)

다음 group 기준:
  K = K32
  Rt = Identity

따라서 picture마다 K/Rt 한 쌍만 저장하는 단순 구조로는 부족할 수 있다.

Geometry group ID 또는 previous/current group state를 구분해서 저장할 수 있는 구조를 설계해라.

K를 picture마다 복제할 필요는 없으며, group state에 K를 보관하고 각 pose가 어느 group 기준인지 식별할 수 있으면 된다.

3. FrameSkip 처리

FrameSkip은 항상 32의 배수이고, encoding 시작 codec POC는 항상 0이다.

Encoder의 camera/depth 입력 인덱스에는 FrameSkip offset을 적용한다.

개념적으로는 다음과 같다.

sourceGeometryIndex = FrameSkip + codecPoc

실제 geometry JSONL과 depth YUV의 indexing 방식이 GOP index, local POC, depth frame index 등을 사용하는 경우 현재 코드 구조를 분석하여 정확한 mapping을 계획해라.

이 mapping은 encoder 입력 처리에만 사용한다.

FrameSkip 값이나 source frame index는 bitstream으로 signaling하지 않는다.

Decoder에는 FrameSkip 전용 상태나 분기를 추가하지 않는다.

Codec POC 0에서 항상 새로운 geometry sequence가 시작된다.

codec POC = 0
groupIdx = 0
anchorPoc = 0
K = FrameSkip 위치 source frame의 K
Rt = Identity

FrameSkip이 32의 배수가 아니면 encoder initialization 단계에서 명확한 오류를 발생시켜라.

현재 코드베이스의 실제 error/check macro를 사용한다.

개념적으로는 다음 조건이다.

CHECK(frameSkip % CAM_GEOMETRY_RESET_PERIOD != 0,
      "FrameSkip must be a multiple of camera geometry reset period");

4. Encoder의 tool on/off 판정

Picture별 tool on/off 판단은 codec의 실제 coding order와 reference 구조를 따라 수행한다.

RA에서는 hierarchical coding order대로 판단한다.

각 picture에 대해 다음을 수행한다.

L0 reference 중 camera projection reference로 사용 가능한 picture를 찾는다.

그중 temporal distance가 가장 가까운 reference를 선택한다.

L1에 대해서도 동일하게 수행한다.

이미 tool off로 결정된 picture는 camera projection reference 후보에서 제외한다.

선택된 L0/L1 reference 각각에 대해 camera parameter와 GT depth를 이용해 대표 pixel motion을 계산한다.

기존 threshold 방식으로 각 방향의 tool 사용 가능 여부를 판정한다.

최종 picture tool flag 또는 list별 availability를 결정한다.

판단은 coding order로 수행하지만, 기록할 metadata는 display POC order로 정렬한다.

기존 motion 판정 방식은 현재 구현을 유지한다.

개념적으로는 다음 값이다.

motion4096 =
    avgMvPx / ((width + height) / 2.0) * 4096

L0/L1의 가장 가까운 reference 각각에 대해 motion을 계산하고, 기존 설정에 따라 min, mean 또는 max 방식으로 결합한다.

finalMotion < threshold → tool off
finalMotion >= threshold → tool on

유효한 projection point가 부족한 경우 현재 구현의 안전한 정책을 유지한다.

예를 들어 충분한 유효 point가 없으면 tool on을 유지하는 방식이 기존 정책이라면 그대로 적용한다.

5. Tool-off reference 제외 규칙

Picture가 tool off로 결정되면 이후 picture의 camera projection reference 후보에서 제외한다.

이 판정은 coding order로 진행한다.

예를 들어 어떤 picture가 먼저 coding되어 tool off로 결정되었다면, 이후 coding되는 picture는 해당 picture를 camera reference로 선택하지 않는다.

Tool-off picture는 다음처럼 처리한다.

camera pose syntax 생략
poseAvailable = false
camProj reference 후보에서 제외
camProj merge 후보에서 제외
camProj regular mode reference에서 제외
camProj skip mode reference에서 제외

단, 기존 일반 inter prediction의 reference picture로 사용하는 것까지 막아서는 안 된다.

즉 DPB에는 존재하지만 camera geometry pose만 unavailable인 상태를 구분해야 한다.

Encoder는 GOP coding order를 따라 tool 여부를 확정한 뒤, 최종 tool flag와 Rt payload를 display order로 재정렬한다.

6. Anchor 및 Intra picture 처리

Intra picture라고 무조건 tool on으로 강제하지 마라.

Geometry group 시작 anchor는 해당 group 내부에서 다음 상태를 가진다.

K = 새 group K
Rt = Identity
poseAvailable = true

하지만 동일한 경계 picture가 이전 group endpoint로 사용될 때는 이전 group 관점의 tool on/off 판단 대상이 될 수 있다.

예를 들어 POC 32가 I slice인 경우:

새 group 기준:
  K32 전송
  Rt = Identity

이전 group 기준:
  POC 32를 camera reference로 필요로 하는 picture가 있는지 판단

POC 32를 가장 가까운 camera reference로 사용하게 될 picture들이 모두 tool off라면, 이전 group 기준 POC 32 pose는 생략 가능하다.

반대로 tool-on picture가 이전 group 기준 POC 32를 camera reference로 사용한다면 해당 endpoint pose는 반드시 available이어야 한다.

이 의존성은 encoder가 coding order로 tool 여부를 판정하면서 자연스럽게 해결하도록 설계해라.

7. L0/L1 availability

Picture별 tool 판정은 L0/L1 각각 가장 가까운 camera-eligible reference를 기준으로 수행한다.

현재 codec의 camProj tool 구조를 분석해서 다음 중 최소 변경안을 선택해라.

A. picture-level 단일 tool flag
B. L0/L1별 camProj availability flag
C. picture-level posePresent와 L0/L1별 camProj availability 분리

Rt 자체는 picture의 pose이므로 L0/L1마다 중복 전송하지 않는다.

최소한 다음 관계는 유지해야 한다.

posePresent =
    camProjL0Available || camProjL1Available

현재 camProj tool이 list별 availability를 지원하지 않고 picture-level enable만 사용하는 구조라면, 기존 구조를 크게 깨지 않는 최소 변경안을 제안해라.

8. Rt 차분 전송 방식

각 geometry group에서 tool-on picture의 Rt만 전송한다.

Tool-off picture의 Rt는 생략한다.

Rt 차분은 직전 display POC가 아니라, 같은 group 내에서 마지막으로 pose가 available했던 tool-on picture를 기준으로 계산한다.

예:

POC 0: group anchor, Identity
POC 1: ON
POC 2: OFF
POC 3: OFF
POC 4: ON

POC 4에서 전송하는 Rt delta는 다음 기준이다.

deltaRt(4) = relativeRt(reconstructedRt(1), originalRt(4))

POC 2 또는 POC 3을 기준으로 계산하면 안 된다.

Decoder는 동일하게 마지막 available pose를 기준으로 복원한다.

lastAvailablePose = groupAnchorIdentity;

for each entry in display order
{
  if (!toolEnabled)
  {
    poseAvailable[poc] = false;
  }
  else
  {
    deltaRt = parseAndDequantizeDeltaRt();
    absoluteRt[poc] = compose(lastAvailablePose, deltaRt);
    poseAvailable[poc] = true;
    lastAvailablePose = absoluteRt[poc];
    lastAvailablePosePoc = poc;
  }
}

9. Encoder reconstructed pose state

Encoder는 original floating-point Rt만 사용해서 다음 Rt delta를 계산하면 안 된다.

Encoder 내부에서도 decoder와 동일한 reconstructed pose state를 유지해야 한다.

순서는 다음과 같다.

1. original target Rt와 encoder reconstructed lastAvailablePose로 delta 계산
2. delta quantization
3. delta dequantization
4. reconstructed absolute Rt 누적
5. 이 reconstructed Rt를 다음 picture의 predictor로 사용

즉 다음 picture의 delta는 original 이전 pose가 아니라 양자화 오차가 반영된 reconstructed pose를 기준으로 계산한다.

Encoder와 decoder는 다음 항목이 완전히 동일해야 한다.

rotation representation
relative pose 방향
matrix multiplication 순서
translation composition
quantization
dequantization
lastAvailablePose update

현재 camera convention이 다음 중 무엇인지 실제 코드를 분석해 정확히 정리해라.

camera_from_world
world_from_camera
current_from_reference
reference_from_current

Rodrigues vector를 직접 차분할지, rotation matrix relative transform 후 다시 rvec으로 변환할지도 현재 구현을 분석해서 결정해라.

단순 rvec component subtraction이 기존 convention과 맞지 않는다면 올바른 relative rotation을 사용해라.

10. Display-order payload 구성

Encoder의 tool on/off 판정은 coding order로 수행한다.

하지만 bitstream payload에 기록되는 entry는 display POC order여야 한다.

각 entry는 개념적으로 다음 정보를 가진다.

entryPoc
toolEnabled 또는 posePresent
optional quantized delta Rt
optional L0/L1 availability

POC 자체를 entry마다 전송할 필요가 있는지 검토해라.

Decoder가 현재 POC와 lastLoadedGeometryPoc를 통해 연속 display-order entry를 복원할 수 있다면, entry POC를 별도 signaling하지 않는 최소 syntax를 우선 고려해라.

11. RA batch signaling

Hierarchical RA에서는 미래 anchor picture가 중간 display-order picture보다 먼저 decoding된다.

예:

display order:
0, 1, 2, ..., 31, 32

decoding order:
0, 32, 16, 8, 4, ...

POC 32를 decoding할 때 POC 1~32의 geometry metadata를 한꺼번에 POC 32의 picture/slice header에 기록한다.

기록 순서는 반드시 display order다.

POC 1 entry
POC 2 entry
...
POC 32 entry

Decoder는 다음 상태를 유지한다.

int lastLoadedGeometryPoc;

현재 picture의 POC까지 아직 geometry metadata가 로드되지 않았다면 반복해서 읽는다.

개념적으로는 다음과 같다.

while (lastLoadedGeometryPoc < currentPoc)
{
  ++lastLoadedGeometryPoc;

  toolEnabled = parseToolFlag();

  if (toolEnabled)
  {
    deltaRt = parseDeltaRt();
    reconstructAbsoluteRt(lastLoadedGeometryPoc, deltaRt);
    markPoseAvailable(lastLoadedGeometryPoc);
  }
  else
  {
    markPoseUnavailable(lastLoadedGeometryPoc);
  }
}

이미 다음 조건이면 geometry payload를 다시 읽지 않는다.

lastLoadedGeometryPoc >= currentPoc

따라서 POC 32에서 POC 1~32 정보를 한 번 로드한 뒤, 이후 POC 16이나 POC 8을 decoding할 때는 기존 geometry state를 조회만 한다.

12. LDB signaling

LDB에서는 coding order와 display order가 순차적이다.

따라서 current picture를 만날 때마다 해당 picture의 tool flag와 optional Rt를 바로 기록한다.

예:

POC 0:
  K
  Identity Rt initialization

POC 1:
  tool flag
  optional delta Rt

POC 2:
  tool flag
  optional delta Rt

Decoder의 기본 loading 로직은 RA와 공통으로 유지할 수 있다.

while (lastLoadedGeometryPoc < currentPoc)

LDB에서는 일반적으로 한 번만 반복된다.

RA/LDB를 별도의 완전히 다른 decoder 구현으로 만들기보다는 공통 metadata state와 loading 함수를 사용하는 방향을 우선 고려해라.

13. K signaling과 reset 순서

각 32-picture geometry group 시작에서 해당 group의 K를 전송한다.

POC 0, 32, 64 등에서 새 K를 전달한다.

새 group 시작 시 decoder는 다음을 수행한다.

1. 새 group K parsing
2. group anchor pose를 Identity로 등록
3. poseAvailable(anchorPoc) = true
4. group의 lastAvailablePose를 Identity로 초기화
5. lastAvailablePosePoc = anchorPoc

RA에서 POC 32 같은 경계 picture header에는 다음 두 정보가 같이 존재할 수 있다.

A. 이전 group의 POC 1~32 metadata
B. 다음 group의 K와 identity anchor 초기화

권장 parsing 순서는 다음과 같다.

1. 이전 group의 아직 load되지 않은 entry를 current POC까지 parsing
2. current POC가 reset boundary이면 다음 group K parsing
3. 다음 group anchor pose를 Identity로 초기화

이 순서라야 POC 32의 이전 group endpoint pose를 먼저 복원하고, 이후 다음 group anchor identity state를 별도로 생성할 수 있다.

단, 현재 slice/picture header syntax 호출 순서상 다른 순서가 더 안전하다면 정확한 이유와 함께 대안을 제시해라.

이전 group endpoint state를 새 group identity state로 덮어쓰면 안 된다.

14. Decoder geometry state

Decoder는 geometry group별 state를 유지해야 한다.

개념적으로 최소 다음 정보가 필요하다.

struct CamGeometryGroupState
{
  int anchorPoc;

  bool kAvailable;
  CameraIntrinsic K;

  std::array<bool, CAM_GEOMETRY_RESET_PERIOD + 1>
      poseAvailable;

  std::array<CameraPose, CAM_GEOMETRY_RESET_PERIOD + 1>
      absolutePose;

  CameraPose lastAvailablePose;
  int lastAvailablePosePoc;
};

실제 코덱 타입과 메모리 구조에 맞게 수정해라.

고정 배열, vector, map, DPB picture 내부 저장 중 어떤 방식이 적절한지 현재 코드의 picture lifetime과 reference lookup 구조를 분석해서 결정해라.

경계 picture가 previous/current 두 group에 속할 수 있으므로 pose lookup 시 최소 다음 정보가 필요하다.

picture POC
geometry group index
해당 group 기준 pose
해당 group의 K

CamProj projection 함수에 target/reference picture만 전달하는 현재 구조라면, 어떤 group state를 선택할지 명확한 규칙을 추가해야 한다.

15. Target/reference group 선택

Camera projection 시 target과 reference가 동일한 geometry group에 속하는지 확인해야 한다.

가능하면 동일 group의 K와 Rt를 사용한다.

경계 picture는 두 group state를 가질 수 있으므로, projection 대상 pair의 POC 범위에 따라 적절한 group을 선택한다.

예:

target POC 20, reference POC 32
→ group 0 기준 POC 32 endpoint pose 사용

target POC 40, reference POC 32
→ group 1 기준 POC 32 anchor identity 사용

실제 camProj reference 선택과 projection 함수에서 이 group 선택을 어디서 수행할지 구체적으로 계획해라.

Target과 reference가 서로 다른 reset group에 있고 공통 geometry 표현이 없는 경우 camera projection reference 후보에서 제외하는 정책을 우선 고려해라.

16. Slice 또는 picture header syntax

현재 camera parameter syntax가 picture header인지 slice header인지 실제 코드를 추적해라.

다음을 확인해라.

encoder write 함수
decoder read 함수
해당 함수 호출 횟수
multi-slice picture에서 반복 여부
picture header 사용 여부
slice header 사용 여부

Geometry metadata는 picture당 한 번만 실질적으로 적용되어야 한다.

현재 설계에서는 decoder가 다음 조건을 통해 중복 parsing을 피할 수 있다.

lastLoadedGeometryPoc < currentPoc

하지만 encoder의 write 조건과 decoder의 read 조건이 반드시 완전히 같아야 한다.

단순히 decoder가 중복 값을 무시하는 방식이면 안 된다.

Syntax가 실제로 존재하는 조건 자체가 encoder와 decoder에서 동일해야 entropy parsing alignment가 유지된다.

확인할 항목:

첫 번째 slice에만 payload를 쓰는지
모든 slice가 같은 조건식을 사용하는지
currentPoc와 lastLoadedGeometryPoc 상태가 slice별로 동일한지
한 picture 내 slice 간 state update 시점이 안전한지

필요하면 picture-level geometry parsing 완료 flag를 추가하는 계획을 제안해라.

17. Decoder 초기화

Codec POC 0에서 geometry state를 초기화한다.

FrameSkip 여부와 관계없이 decoder가 보는 첫 picture는 항상 POC 0이다.

POC 0에서 다음을 수행한다.

1. 이전 sequence의 geometry state 제거
2. group 0 K parsing
3. POC 0 pose를 Identity로 등록
4. poseAvailable(0) = true
5. lastAvailablePose = Identity
6. lastAvailablePosePoc = 0
7. lastLoadedGeometryPoc = 0

Decoder에는 FrameSkip 값이나 source frame offset을 저장하지 않는다.

18. IDR, CRA, POC reset

현재 실험의 기본 전제는 codec POC 0부터 geometry state가 시작하는 것이다.

그럼에도 실제 코드에서 다음 lifecycle을 분석해라.

새 sequence 시작
IDR
CRA
POC reset
decoder flush
random access
parameter set 재초기화

Geometry state가 언제 초기화되어야 하는지 실제 decoder 호출 흐름을 기준으로 정리해라.

IntraPeriod마다 무조건 geometry state를 초기화하면 안 된다.

Geometry reset은 codec POC 기준 매 32 picture마다 수행하며, IntraPeriod와 독립적이다.

다만 IDR로 codec POC가 다시 0부터 시작하는 구조라면 새 geometry sequence로 초기화한다.

CRA에서 POC가 유지되는 경우 기존 group state를 유지할지, random access 진입을 위해 새 K/anchor state가 필요한지 현재 코덱의 random access 동작을 분석해 계획에 포함해라.

19. Incomplete final group

Sequence 마지막 geometry group은 32 picture보다 짧을 수 있다.

예:

마지막 POC = 75
마지막 group anchor = 64

이 경우 존재하는 picture까지만 metadata를 기록한다.

없는 POC까지 dummy entry를 기록하지 않는다.

RA batch payload의 entry 개수는 현재 picture POC와 lastLoadedGeometryPoc 차이 또는 실제 존재하는 picture 범위로 결정해야 한다.

Sequence 끝에서 encoder와 decoder가 payload entry 개수를 동일하게 판단할 수 있도록 syntax 조건을 명확히 정리해라.

20. 기존 camProj tool과 연결

현재 코드에서 다음 호출 경로를 실제로 찾아라.

camera JSON 또는 parameter file parsing
depth YUV parsing
FrameSkip 적용 위치
picture POC 확정 위치
picture별 K/Rt 저장 구조
RPL 생성 시점
L0/L1 reference 확정 시점
camProj picture-level enable 판정
camProj merge candidate 생성
camProj regular prediction
camProj skip mode
projection MV 생성
spanMotionInfo 저장
DPB picture lookup
Picture 객체 생성 및 삭제
slice/picture header write
slice/picture header read
encoder reconstructed picture state
decoder reconstructed picture state

Tool-off reference가 다음 모든 경로에서 제외되는지 각각 확인해라.

camProj merge
camProj merge skip
camProj regular
camProj reference candidate list
camera-based MV generation
camera-based affine candidate
HMVP에 camera candidate를 추가하는 경로가 있다면 해당 경로

기존 일반 merge, skip, AMVP, affine, inter prediction에는 영향을 주지 않아야 한다.

21. Encoder buffering 구조

RA에서는 picture별 metadata를 coding order로 판단한 뒤 display order로 저장할 buffer가 필요하다.

개념적으로 다음 정보가 필요하다.

struct CamGeometryPendingEntry
{
  int poc;

  bool toolEnabled;
  bool posePresent;

  bool l0Available;
  bool l1Available;

  QuantizedCameraDeltaRt deltaRt;
};

실제 codec 구조에 맞는 타입을 사용해라.

Metadata가 어느 시점에 확정되는지 분석해라.

picture encode 전
RPL 확정 후
camera motion threshold 평가 후
picture header write 전
GOP compression 전체 완료 후

RA에서는 POC 32 header를 쓸 때 POC 1~32의 decision이 모두 이미 존재해야 한다.

현재 encoder pipeline에서 POC 32 header가 언제 작성되고, POC 1~31의 tool decision이 그 시점에 이미 계산 가능한지 반드시 확인해라.

만약 현재 pipeline상 POC 32 header를 쓰는 시점에 중간 picture decision이 아직 계산되지 않았다면, 다음 대안을 비교해라.

GOP 단위 pre-analysis 수행
camera tool on/off만 별도 선행 분석
header write 지연
별도 metadata NAL 사용

가장 적은 구조 변경으로 가능한 방식을 제안해라.

이 부분은 구현 가능성을 결정하는 핵심이므로 상세히 분석해라.

22. Decoder loading pointer

Decoder는 최소 다음 상태를 가진다.

int lastLoadedGeometryPoc;

하지만 group boundary dual state 때문에 필요하다면 다음처럼 group별 pointer 또는 현재 active group 상태를 추가할 수 있다.

currentLoadGroupIdx
lastLoadedGeometryPoc
previousGroupState
currentGroupState

단순히 하나의 전역 pointer로 충분한지 실제 POC reset과 group state lifetime을 분석해 결정해라.

Pointer 업데이트는 metadata entry parsing이 정상 완료된 뒤에 수행해야 한다.

Parsing 중 오류가 발생했거나 필요한 delta 값이 부족한 상태에서 pointer만 먼저 증가하면 안 된다.

23. Encoder/decoder 대칭성

다음 조건에 대해 encoder write와 decoder read를 함수 단위로 대칭시켜라.

K 존재 조건
K quantization
K write/read 순서
tool flag write/read 조건
L0/L1 availability write/read 조건
posePresent 조건
Rt delta write/read 조건
rotation quantization
translation quantization
relative Rt 계산
absolute Rt reconstruction
lastAvailablePose 갱신
lastAvailablePosePoc 갱신
lastLoadedGeometryPoc 갱신
group reset
boundary dual state 생성
pose unavailable 처리
FrameSkip input mapping
POC 0 초기화

계획에서 encoder 함수와 대응 decoder 함수를 나란히 명시해라.

24. Debug dump

#if GlobalMotion 내부에 encoder와 decoder debug dump 기능을 추가하는 계획을 포함해라.

CSV 출력은 compile-time 또는 runtime debug option으로 켤 수 있게 한다.

Encoder와 decoder의 권장 컬럼:

codingOrder
currentPoc
payloadEntryPoc
geometryGroupIdx
geometryAnchorPoc
entryType
toolEnabled
posePresent
l0Available
l1Available
lastAvailablePosePocBefore
quantizedR0
quantizedR1
quantizedR2
quantizedT0
quantizedT1
quantizedT2
reconstructedR0
reconstructedR1
reconstructedR2
reconstructedT0
reconstructedT1
reconstructedT2
lastAvailablePosePocAfter
lastLoadedGeometryPocBefore
lastLoadedGeometryPocAfter
poseAvailable

Entry type 예:

K_RESET
POSE_ON
POSE_OFF
BOUNDARY_ENDPOINT
BOUNDARY_ANCHOR

Encoder와 decoder CSV를 POC, group, entry type 기준으로 비교해서 mismatch를 검출할 수 있게 해라.

Floating-point reconstructed Rt 비교는 허용 오차와 함께 수행하되, quantized syntax 값은 bit-exact하게 동일해야 한다.

25. 검증 시나리오

최소 다음 설정을 검증하도록 계획해라.

1. RA, GOP 32, IntraPeriod 32
2. RA, GOP 32, IntraPeriod 64
3. RA, GOP 16, IntraPeriod 64
4. RA, GOP 64, IntraPeriod 64
5. LDB, GOP 32
6. LDB, GOP 16
7. 연속된 여러 picture가 tool off
8. tool off 뒤 다시 tool on
9. reset boundary 이전 group endpoint가 tool on
10. reset boundary 이전 group endpoint가 tool off
11. reset boundary가 I slice
12. reset boundary가 B slice
13. 마지막 group이 32 picture보다 짧음
14. multi-slice picture
15. FrameSkip 0
16. FrameSkip 32
17. FrameSkip 64
18. FrameSkip 96
19. FrameSkip이 1, 10, 31, 33일 때 encoder가 오류 발생
20. encoder/decoder reconstructed Rt 완전 일치
21. tool-off picture가 모든 camProj candidate에서 제외됨
22. 일반 inter reference 사용에는 영향 없음

RA에서는 특히 다음 decoding order를 검증해라.

0 → 32 → 16 → 8 → ...

POC 32에서 POC 1~32 metadata가 한 번 로드되고, 이후 POC 16, 8 등에서 재로딩되지 않는지 확인한다.

26. 구현 계획 결과물 형식

다음 순서로 implementation plan을 작성해라.

1) 현재 코드 구조 분석

관련 파일명

관련 클래스와 구조체

camera/depth 입력 흐름

FrameSkip 처리 흐름

encoder POC 생성 흐름

RA/LDB coding order 흐름

RPL 생성 흐름

picture/slice header write 위치

decoder header read 위치

기존 camProj tool 호출 흐름

2) 현재 구조에서 구현 가능성 분석

특히 다음을 확인해라.

POC 32 header 작성 전에 POC 1~31 tool decision을 계산할 수 있는지

별도 GOP pre-analysis가 필요한지

기존 header syntax에 batch metadata를 넣을 수 있는지

picture마다 두 geometry group state를 저장할 수 있는지

3) 제안 bitstream syntax

각 syntax element에 대해 다음을 작성해라.

syntax 이름
의미
write 조건
read 조건
coding 방식
반복 횟수
RA/LDB 차이
reset boundary 처리

4) Encoder 변경 계획

FrameSkip source index mapping

FrameSkip validation

GOP pre-analysis

coding-order tool 판정

tool-off reference 제외

display-order buffering

K quantization

Rt delta quantization

reconstructed pose shadow state

RA batch write

LDB immediate write

final short group 처리

5) Decoder 변경 계획

POC 0 초기화

geometry group state

lastLoadedGeometryPoc

display-order parsing loop

K reset

boundary dual state

tool-off pose unavailable 처리

Rt 누적 복원

camProj reference lookup

IDR/CRA/POC reset 처리

multi-slice 중복 방지

6) 기존 camProj tool 연결

각 모드별로 실제 파일과 함수를 명시해라.

camProj merge
camProj merge skip
camProj regular
camera MV generation
affine camera candidate
reference candidate filtering
DPB lookup

7) 수정 파일 및 함수 목록

반드시 실제 코드에 존재하는 파일명과 함수명으로 작성해라.

각 함수별로 다음을 작성한다.

현재 역할
추가할 상태 또는 인자
추가할 로직
호출 순서 변경
encoder/decoder 대응 함수

8) Debug 및 mismatch 검증 계획

encoder CSV

decoder CSV

quantized syntax 비교

reconstructed Rt 비교

pointer 비교

group state 비교

자동 mismatch 검출 방법

9) 위험 요소와 해결책

최소 다음을 포함한다.

RA header 작성 시점
FrameSkip input indexing
boundary picture dual state
rotation composition convention
quantization drift
multi-slice parsing
POC reset
IDR/CRA
incomplete final group
memory lifetime
tool-off reference 누락

10) 단계별 구현 순서

안전한 구현 순서를 제안해라.

권장 예:

1. geometry state 자료구조
2. FrameSkip input mapping
3. encoder pre-analysis
4. display-order buffer
5. syntax write/read
6. decoder reconstruction
7. camProj reference filtering
8. debug dump
9. RA 검증
10. LDB 검증

아직 실제 코드를 변경하지 마라.

먼저 코드베이스를 충분히 탐색하고, 위 요구사항을 실제 코드 구조에 매핑한 상세 implementation plan만 작성해라.

코드에서 직접 확인할 수 있는 내용은 나에게 질문하지 말고 직접 분석해서 결정해라.

구현 전에 반드시 사용자의 설계 선택이 필요한 항목만 마지막에 별도로 정리해라.
