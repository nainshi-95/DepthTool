#ifndef DEPTH_CAM_PARAM_H
#define DEPTH_CAM_PARAM_H

#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <string>

namespace depthcam
{

struct DepthCamParamConfig
{
  int    predN         = 3;
  int    predDegree    = 2;     // internally clamped to [0, 2]

  int    extBits       = 8;     // signed symmetric range: [-(2^(bits-1)-1), +(2^(bits-1)-1)]
  double rStep         = 1.0 / 4096.0; // 2^-12
  double tStepNorm     = 1.0 / 1024.0; // 2^-10, for t / depthScale

  int    maxPredHist   = 8;     // history size used by predictor
  int    maxFrameStore = 64;    // history size used for curPoc -> refPoc transform composition

  double intrFMax      = 4.0;   // fx/w, fy/h quant range: [-intrFMax, +intrFMax]
  double intrCMin      = -1.0;  // cx/w, cy/h quant range
  double intrCMax      = 2.0;
};

struct DepthCamIntrinsic
{
  double fx    = 0.0;
  double fy    = 0.0;
  double cx    = 0.0;
  double cy    = 0.0;
  double zSign = -1.0;
};

struct DepthCamHeader
{
  double depthScale = 1.0;

  // Quantized intrinsic syntax. Decoder should receive these values from bitstream/header.
  uint16_t qFx = 0;
  uint16_t qFy = 0;
  uint16_t qCx = 0;
  uint16_t qCy = 0;

  // Sign convention for camera forward axis. Usually -1.0 for OpenGL-like camera coordinates.
  double zSign = -1.0;

  // Encoder-side information/debug. Decoder recomputes this from qFx/qFy/qCx/qCy.
  DepthCamIntrinsic intrinsicGt;
  DepthCamIntrinsic intrinsicDec;

  bool intrinsicClipped = false;
};

struct DepthCamParam6
{
  // 0: rx
  // 1: ry
  // 2: rz
  // 3: tx / depthScale
  // 4: ty / depthScale
  // 5: tz / depthScale
  std::array<double, 6> v{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
};

struct DepthCamQResidual
{
  // Quantized residual of Param6.
  // q[0:3] use rStep.
  // q[3:6] use tStepNorm.
  std::array<int, 6> q{{0, 0, 0, 0, 0, 0}};
  bool clipped = false;
};

struct DepthCamProjectionParam
{
  int curPoc = -1;
  int refPoc = -1;

  // Intrinsic used by both encoder and decoder. This must be dequantized intrinsic.
  DepthCamIntrinsic intr;

  // Current camera coordinate -> reference camera coordinate.
  // Column-vector convention:
  //   P_ref = R_curToRef * P_cur + t_curToRef
  // Row-major 3x3 matrix.
  std::array<double, 9> R{{1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0}};
  std::array<double, 3> t{{0.0, 0.0, 0.0}};

  double depthScale = 1.0;
};

class DepthCamParamBase
{
public:
  virtual ~DepthCamParamBase() = default;

  bool isValid() const { return m_valid; }
  const std::string& getLastError() const { return m_lastError; }

  int getWidth() const { return m_width; }
  int getHeight() const { return m_height; }

  const DepthCamParamConfig& getConfig() const { return m_cfg; }
  const DepthCamHeader& getHeader() const { return m_header; }

  int getStoredFrameCount() const { return static_cast<int>(m_frameStore.size()); }

  // Returns cur camera -> ref camera transform for backward projection.
  // This requires all adjacent transforms between curPoc and refPoc to be stored.
  bool getProjectionParam(int curPoc, int refPoc, DepthCamProjectionParam& out) const;

  // Optional helper for pixel-level backward projection.
  // depthLinear must already be physical linear depth, e.g. depthY * depthScale.
  static bool mapCurPixelToRef(
      const DepthCamProjectionParam& proj,
      double curX,
      double curY,
      double depthLinear,
      double& refX,
      double& refY);

  static int signedQAbsMax(int bits);

protected:
  DepthCamParamBase(int width, int height, const DepthCamParamConfig& cfg);

  bool initHeaderFromIntrinsic(double depthScale, const DepthCamIntrinsic& intrinsicGt);
  bool initHeaderFromQuant(const DepthCamHeader& headerWithQuant);

  bool encodeAndStore(int poc, const DepthCamParam6& gtParam, DepthCamQResidual& outQ);
  bool setAndStore(int poc, const DepthCamQResidual& q);

  DepthCamParam6 predictFromHistory() const;

  void clearHistory();

  void setError(const std::string& err) const;

private:
  struct FrameRecord
  {
    int poc = -1;
    bool hasAdjacent = false; // true if this record contains T[poc -> poc-1]

    DepthCamParam6 decParam;

    // Adjacent transform T[poc -> poc-1]. Row-major R.
    std::array<double, 9> R{{1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0}};
    std::array<double, 3> t{{0.0, 0.0, 0.0}};
  };

  FrameRecord makeRecord(int poc, const DepthCamParam6& decParam) const;
  bool pushRecord(const FrameRecord& rec);
  const FrameRecord* findRecord(int poc) const;

  bool validateQResidual(const DepthCamQResidual& q) const;

  static bool solveSmallLinearSystem(const double* A, const double* b, int n, double* x);

  static void setIdentity(std::array<double, 9>& R);
  static void setZero(std::array<double, 3>& t);

  static void rodriguesToMatrix(const double rvec[3], std::array<double, 9>& R);

  static void matMul3x3(
      const std::array<double, 9>& A,
      const std::array<double, 9>& B,
      std::array<double, 9>& C);

  static void matVecMul3(
      const std::array<double, 9>& A,
      const std::array<double, 3>& x,
      std::array<double, 3>& y);

  static void composeLeft(
      const std::array<double, 9>& RLeft,
      const std::array<double, 3>& tLeft,
      std::array<double, 9>& RAcc,
      std::array<double, 3>& tAcc);

  static void inverseRigid(
      const std::array<double, 9>& R,
      const std::array<double, 3>& t,
      std::array<double, 9>& RInv,
      std::array<double, 3>& tInv);

  static uint16_t quantU16(double value, double lo, double hi, bool& clipped);
  static double dequantU16(uint16_t q, double lo, double hi);

  static int quantS(double value, double step, int bits, bool& clipped);
  static double dequantS(int q, double step);

protected:
  int m_width = 0;
  int m_height = 0;

  DepthCamParamConfig m_cfg;
  DepthCamHeader m_header;

  bool m_valid = false;
  mutable std::string m_lastError;

private:
  std::deque<FrameRecord> m_predHist;
  std::deque<FrameRecord> m_frameStore;
};

class DepthCamParamEncoder : public DepthCamParamBase
{
public:
  DepthCamParamEncoder(
      const std::string& jsonlPath,
      int width,
      int height,
      const DepthCamParamConfig& cfg = DepthCamParamConfig());

  ~DepthCamParamEncoder();

  // Sequential load. poc must increase by 1 externally if intermediate POCs are needed.
  // For poc == 0, zero parameter is inserted and no frame line is consumed.
  bool load(int poc, DepthCamQResidual& outQ);

private:
  struct JsonFrame
  {
    int poc = -1;
    std::array<double, 3> rvec{{0.0, 0.0, 0.0}};
    std::array<double, 3> tvec{{0.0, 0.0, 0.0}};
  };

  bool readHeaderFromJsonl();
  bool readNextFrame(JsonFrame& frame);

  static bool parseHeaderLine(const std::string& line, double& depthScale, DepthCamIntrinsic& intr);
  static bool parseFrameLine(const std::string& line, JsonFrame& frame);

  std::string m_jsonlPath;
  std::ifstream m_fin;
};

class DepthCamParamDecoder : public DepthCamParamBase
{
public:
  DepthCamParamDecoder(
      const DepthCamHeader& headerFromBitstream,
      int width,
      int height,
      const DepthCamParamConfig& cfg = DepthCamParamConfig());

  // Insert one decoded frame parameter from quantized residual syntax.
  // Must be called in increasing POC order, including intermediate POCs if they are needed.
  bool set(int poc, const DepthCamQResidual& q);
};

} // namespace depthcam

#endif // DEPTH_CAM_PARAM_H
