#include "DepthCamParam.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

namespace depthcam
{
namespace
{
constexpr double kEps = 1e-12;

bool isBlankLine(const std::string& s)
{
  for (char c : s)
  {
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n')
    {
      return false;
    }
  }
  return true;
}

bool findKey(const std::string& line, const std::string& key, std::size_t& keyPos)
{
  const std::string pat = "\"" + key + "\"";
  keyPos = line.find(pat);
  return keyPos != std::string::npos;
}

bool parseDoubleAfterKey(const std::string& line, const std::string& key, double& value)
{
  std::size_t keyPos = 0;
  if (!findKey(line, key, keyPos))
  {
    return false;
  }

  std::size_t colon = line.find(':', keyPos);
  if (colon == std::string::npos)
  {
    return false;
  }

  const char* begin = line.c_str() + colon + 1;
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(begin, &end);
  if (begin == end || errno == ERANGE)
  {
    return false;
  }

  value = v;
  return true;
}

bool parseIntAfterKey(const std::string& line, const std::string& key, int& value)
{
  double d = 0.0;
  if (!parseDoubleAfterKey(line, key, d))
  {
    return false;
  }
  value = static_cast<int>(std::llround(d));
  return true;
}

bool parseStringAfterKey(const std::string& line, const std::string& key, std::string& value)
{
  std::size_t keyPos = 0;
  if (!findKey(line, key, keyPos))
  {
    return false;
  }

  std::size_t colon = line.find(':', keyPos);
  if (colon == std::string::npos)
  {
    return false;
  }

  std::size_t q0 = line.find('"', colon + 1);
  if (q0 == std::string::npos)
  {
    return false;
  }

  std::size_t q1 = line.find('"', q0 + 1);
  if (q1 == std::string::npos || q1 <= q0)
  {
    return false;
  }

  value = line.substr(q0 + 1, q1 - q0 - 1);
  return true;
}

bool parseArray3AfterKey(const std::string& line, const std::string& key, std::array<double, 3>& value)
{
  std::size_t keyPos = 0;
  if (!findKey(line, key, keyPos))
  {
    return false;
  }

  std::size_t lb = line.find('[', keyPos);
  std::size_t rb = line.find(']', lb == std::string::npos ? keyPos : lb);
  if (lb == std::string::npos || rb == std::string::npos || rb <= lb)
  {
    return false;
  }

  const char* p = line.c_str() + lb + 1;
  const char* endLimit = line.c_str() + rb;

  for (int i = 0; i < 3; ++i)
  {
    while (p < endLimit && (*p == ' ' || *p == '\t' || *p == ','))
    {
      ++p;
    }

    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(p, &end);
    if (p == end || errno == ERANGE || end > endLimit)
    {
      return false;
    }

    value[i] = v;
    p = end;
  }

  return true;
}

std::string makeRangeError(const char* name, int q, int qAbsMax)
{
  std::ostringstream oss;
  oss << name << " q=" << q << " outside signed range [-" << qAbsMax << ", " << qAbsMax << "]";
  return oss.str();
}

} // unnamed namespace

// ============================================================================
// Base
// ============================================================================

DepthCamParamBase::DepthCamParamBase(int width, int height, const DepthCamParamConfig& cfg)
  : m_width(width), m_height(height), m_cfg(cfg)
{
  if (m_cfg.predN < 1)         { m_cfg.predN = 1; }
  if (m_cfg.predDegree < 0)    { m_cfg.predDegree = 0; }
  if (m_cfg.predDegree > 2)    { m_cfg.predDegree = 2; }
  if (m_cfg.extBits < 2)       { m_cfg.extBits = 2; }
  if (m_cfg.maxPredHist < 1)   { m_cfg.maxPredHist = 1; }
  if (m_cfg.maxFrameStore < 1) { m_cfg.maxFrameStore = 1; }
}

void DepthCamParamBase::setError(const std::string& err) const
{
  m_lastError = err;
}

int DepthCamParamBase::signedQAbsMax(int bits)
{
  if (bits < 2)
  {
    return 0;
  }
  return (1 << (bits - 1)) - 1;
}

void DepthCamParamBase::clearHistory()
{
  m_predHist.clear();
  m_frameStore.clear();
}

bool DepthCamParamBase::initHeaderFromIntrinsic(double depthScale, const DepthCamIntrinsic& intrinsicGt)
{
  if (m_width <= 0 || m_height <= 0)
  {
    setError("invalid picture size");
    return false;
  }
  if (!(depthScale > 0.0) || !std::isfinite(depthScale))
  {
    setError("invalid depthScale");
    return false;
  }

  bool clipped = false;
  bool c = false;

  const double fxN = intrinsicGt.fx / static_cast<double>(m_width);
  const double fyN = intrinsicGt.fy / static_cast<double>(m_height);
  const double cxN = intrinsicGt.cx / static_cast<double>(m_width);
  const double cyN = intrinsicGt.cy / static_cast<double>(m_height);

  const uint16_t qFx = quantU16(fxN, -m_cfg.intrFMax, m_cfg.intrFMax, c); clipped = clipped || c;
  const uint16_t qFy = quantU16(fyN, -m_cfg.intrFMax, m_cfg.intrFMax, c); clipped = clipped || c;
  const uint16_t qCx = quantU16(cxN,  m_cfg.intrCMin, m_cfg.intrCMax, c); clipped = clipped || c;
  const uint16_t qCy = quantU16(cyN,  m_cfg.intrCMin, m_cfg.intrCMax, c); clipped = clipped || c;

  DepthCamHeader h;
  h.depthScale = depthScale;
  h.qFx = qFx;
  h.qFy = qFy;
  h.qCx = qCx;
  h.qCy = qCy;
  h.zSign = intrinsicGt.zSign;
  h.intrinsicGt = intrinsicGt;
  h.intrinsicClipped = clipped;

  if (!initHeaderFromQuant(h))
  {
    return false;
  }

  m_header.intrinsicGt = intrinsicGt;
  m_header.intrinsicClipped = clipped;
  return true;
}

bool DepthCamParamBase::initHeaderFromQuant(const DepthCamHeader& headerWithQuant)
{
  if (m_width <= 0 || m_height <= 0)
  {
    setError("invalid picture size");
    return false;
  }
  if (!(headerWithQuant.depthScale > 0.0) || !std::isfinite(headerWithQuant.depthScale))
  {
    setError("invalid depthScale in header");
    return false;
  }

  m_header = headerWithQuant;

  const double fxN = dequantU16(m_header.qFx, -m_cfg.intrFMax, m_cfg.intrFMax);
  const double fyN = dequantU16(m_header.qFy, -m_cfg.intrFMax, m_cfg.intrFMax);
  const double cxN = dequantU16(m_header.qCx,  m_cfg.intrCMin, m_cfg.intrCMax);
  const double cyN = dequantU16(m_header.qCy,  m_cfg.intrCMin, m_cfg.intrCMax);

  m_header.intrinsicDec.fx = fxN * static_cast<double>(m_width);
  m_header.intrinsicDec.fy = fyN * static_cast<double>(m_height);
  m_header.intrinsicDec.cx = cxN * static_cast<double>(m_width);
  m_header.intrinsicDec.cy = cyN * static_cast<double>(m_height);
  m_header.intrinsicDec.zSign = (m_header.zSign >= 0.0) ? 1.0 : -1.0;
  m_header.zSign = m_header.intrinsicDec.zSign;

  m_valid = true;
  return true;
}

uint16_t DepthCamParamBase::quantU16(double value, double lo, double hi, bool& clipped)
{
  clipped = false;
  if (hi <= lo)
  {
    clipped = true;
    return 0;
  }

  if (value < lo)
  {
    value = lo;
    clipped = true;
  }
  else if (value > hi)
  {
    value = hi;
    clipped = true;
  }

  const double qMax = 65535.0;
  double q = std::round((value - lo) / (hi - lo) * qMax);
  if (q < 0.0)     { q = 0.0; }
  if (q > qMax)    { q = qMax; }
  return static_cast<uint16_t>(q);
}

double DepthCamParamBase::dequantU16(uint16_t q, double lo, double hi)
{
  const double qMax = 65535.0;
  return static_cast<double>(q) / qMax * (hi - lo) + lo;
}

int DepthCamParamBase::quantS(double value, double step, int bits, bool& clipped)
{
  clipped = false;
  const int qAbsMax = signedQAbsMax(bits);
  if (!(step > 0.0) || qAbsMax <= 0)
  {
    clipped = true;
    return 0;
  }

  long long q64 = std::llround(value / step);
  if (q64 < -qAbsMax)
  {
    q64 = -qAbsMax;
    clipped = true;
  }
  else if (q64 > qAbsMax)
  {
    q64 = qAbsMax;
    clipped = true;
  }

  return static_cast<int>(q64);
}

double DepthCamParamBase::dequantS(int q, double step)
{
  return static_cast<double>(q) * step;
}

bool DepthCamParamBase::validateQResidual(const DepthCamQResidual& q) const
{
  const int qAbsMax = signedQAbsMax(m_cfg.extBits);
  for (int i = 0; i < 6; ++i)
  {
    if (q.q[i] < -qAbsMax || q.q[i] > qAbsMax)
    {
      setError(makeRangeError("extrinsic residual", q.q[i], qAbsMax));
      return false;
    }
  }
  return true;
}

bool DepthCamParamBase::encodeAndStore(int poc, const DepthCamParam6& gtParam, DepthCamQResidual& outQ)
{
  if (!m_valid)
  {
    setError("DepthCamParamBase is not initialized");
    return false;
  }

  const DepthCamParam6 pred = predictFromHistory();
  DepthCamQResidual q;
  q.clipped = false;

  for (int i = 0; i < 3; ++i)
  {
    bool clipped = false;
    q.q[i] = quantS(gtParam.v[i] - pred.v[i], m_cfg.rStep, m_cfg.extBits, clipped);
    q.clipped = q.clipped || clipped;
  }
  for (int i = 3; i < 6; ++i)
  {
    bool clipped = false;
    q.q[i] = quantS(gtParam.v[i] - pred.v[i], m_cfg.tStepNorm, m_cfg.extBits, clipped);
    q.clipped = q.clipped || clipped;
  }

  if (!setAndStore(poc, q))
  {
    return false;
  }

  outQ = q;
  return true;
}

bool DepthCamParamBase::setAndStore(int poc, const DepthCamQResidual& q)
{
  if (!m_valid)
  {
    setError("DepthCamParamBase is not initialized");
    return false;
  }
  if (!validateQResidual(q))
  {
    return false;
  }
  if (!m_frameStore.empty() && poc <= m_frameStore.back().poc)
  {
    std::ostringstream oss;
    oss << "POC must be inserted in strictly increasing order. requested=" << poc
        << ", last=" << m_frameStore.back().poc;
    setError(oss.str());
    return false;
  }

  const DepthCamParam6 pred = predictFromHistory();

  DepthCamParam6 dec;
  for (int i = 0; i < 3; ++i)
  {
    dec.v[i] = pred.v[i] + dequantS(q.q[i], m_cfg.rStep);
  }
  for (int i = 3; i < 6; ++i)
  {
    dec.v[i] = pred.v[i] + dequantS(q.q[i], m_cfg.tStepNorm);
  }

  FrameRecord rec = makeRecord(poc, dec);
  return pushRecord(rec);
}

DepthCamParam6 DepthCamParamBase::predictFromHistory() const
{
  DepthCamParam6 pred;

  if (m_predHist.empty())
  {
    return pred;
  }

  const int available = static_cast<int>(m_predHist.size());
  const int m = std::min(available, std::max(1, m_cfg.predN));

  if (m == 1)
  {
    return m_predHist.back().decParam;
  }

  const int degree = std::min(std::min(m_cfg.predDegree, m - 1), 2);

  if (degree <= 0)
  {
    for (int d = 0; d < 6; ++d)
    {
      double sum = 0.0;
      for (int i = available - m; i < available; ++i)
      {
        sum += m_predHist[i].decParam.v[d];
      }
      pred.v[d] = sum / static_cast<double>(m);
    }
    return pred;
  }

  const int nCoef = degree + 1;

  // Normal equation: (A^T A)c = A^T y, A=[1,x,x^2], x=0..m-1.
  double ATA[9] = {0.0}; // max 3x3, row-major
  for (int row = 0; row < nCoef; ++row)
  {
    for (int col = 0; col < nCoef; ++col)
    {
      double s = 0.0;
      for (int i = 0; i < m; ++i)
      {
        s += std::pow(static_cast<double>(i), row + col);
      }
      ATA[row * nCoef + col] = s;
    }
  }

  const double xNext = static_cast<double>(m);

  for (int d = 0; d < 6; ++d)
  {
    double ATy[3] = {0.0, 0.0, 0.0};

    for (int row = 0; row < nCoef; ++row)
    {
      double s = 0.0;
      for (int i = 0; i < m; ++i)
      {
        const int histIdx = available - m + i;
        s += std::pow(static_cast<double>(i), row) * m_predHist[histIdx].decParam.v[d];
      }
      ATy[row] = s;
    }

    double coef[3] = {0.0, 0.0, 0.0};
    if (!solveSmallLinearSystem(ATA, ATy, nCoef, coef))
    {
      pred.v[d] = m_predHist.back().decParam.v[d];
      continue;
    }

    double y = 0.0;
    double xp = 1.0;
    for (int k = 0; k < nCoef; ++k)
    {
      y += coef[k] * xp;
      xp *= xNext;
    }
    pred.v[d] = y;
  }

  return pred;
}

bool DepthCamParamBase::solveSmallLinearSystem(const double* A, const double* b, int n, double* x)
{
  if (n <= 0 || n > 3)
  {
    return false;
  }

  double M[3][4] = {{0.0}};
  for (int r = 0; r < n; ++r)
  {
    for (int c = 0; c < n; ++c)
    {
      M[r][c] = A[r * n + c];
    }
    M[r][n] = b[r];
  }

  for (int col = 0; col < n; ++col)
  {
    int pivot = col;
    double best = std::fabs(M[col][col]);
    for (int r = col + 1; r < n; ++r)
    {
      const double v = std::fabs(M[r][col]);
      if (v > best)
      {
        best = v;
        pivot = r;
      }
    }

    if (best < 1e-14)
    {
      return false;
    }

    if (pivot != col)
    {
      for (int c = col; c <= n; ++c)
      {
        std::swap(M[col][c], M[pivot][c]);
      }
    }

    const double div = M[col][col];
    for (int c = col; c <= n; ++c)
    {
      M[col][c] /= div;
    }

    for (int r = 0; r < n; ++r)
    {
      if (r == col)
      {
        continue;
      }
      const double f = M[r][col];
      for (int c = col; c <= n; ++c)
      {
        M[r][c] -= f * M[col][c];
      }
    }
  }

  for (int i = 0; i < n; ++i)
  {
    x[i] = M[i][n];
  }
  return true;
}

DepthCamParamBase::FrameRecord DepthCamParamBase::makeRecord(int poc, const DepthCamParam6& decParam) const
{
  FrameRecord rec;
  rec.poc = poc;
  rec.decParam = decParam;

  setIdentity(rec.R);
  setZero(rec.t);

  if (poc <= 0)
  {
    rec.hasAdjacent = false;
    return rec;
  }

  double rvec[3] = { decParam.v[0], decParam.v[1], decParam.v[2] };
  rodriguesToMatrix(rvec, rec.R);

  rec.t[0] = decParam.v[3] * m_header.depthScale;
  rec.t[1] = decParam.v[4] * m_header.depthScale;
  rec.t[2] = decParam.v[5] * m_header.depthScale;

  rec.hasAdjacent = true;
  return rec;
}

bool DepthCamParamBase::pushRecord(const FrameRecord& rec)
{
  m_predHist.push_back(rec);
  while (static_cast<int>(m_predHist.size()) > m_cfg.maxPredHist)
  {
    m_predHist.pop_front();
  }

  m_frameStore.push_back(rec);
  while (static_cast<int>(m_frameStore.size()) > m_cfg.maxFrameStore)
  {
    m_frameStore.pop_front();
  }

  return true;
}

const DepthCamParamBase::FrameRecord* DepthCamParamBase::findRecord(int poc) const
{
  for (auto it = m_frameStore.rbegin(); it != m_frameStore.rend(); ++it)
  {
    if (it->poc == poc)
    {
      return &(*it);
    }
  }
  return nullptr;
}

bool DepthCamParamBase::getProjectionParam(int curPoc, int refPoc, DepthCamProjectionParam& out) const
{
  if (!m_valid)
  {
    setError("DepthCamParamBase is not initialized");
    return false;
  }

  out.curPoc = curPoc;
  out.refPoc = refPoc;
  out.intr = m_header.intrinsicDec;
  out.depthScale = m_header.depthScale;
  setIdentity(out.R);
  setZero(out.t);

  if (curPoc == refPoc)
  {
    return true;
  }

  if (curPoc > refPoc)
  {
    // Compose T[p -> p-1] from p=curPoc down to refPoc+1.
    for (int p = curPoc; p > refPoc; --p)
    {
      const FrameRecord* rec = findRecord(p);
      if (!rec || !rec->hasAdjacent)
      {
        std::ostringstream oss;
        oss << "missing adjacent transform T[" << p << " -> " << (p - 1)
            << "] for getProjectionParam(" << curPoc << ", " << refPoc << ")";
        setError(oss.str());
        return false;
      }
      composeLeft(rec->R, rec->t, out.R, out.t);
    }
    return true;
  }

  // curPoc < refPoc. Use inverse of T[p -> p-1].
  for (int p = curPoc + 1; p <= refPoc; ++p)
  {
    const FrameRecord* rec = findRecord(p);
    if (!rec || !rec->hasAdjacent)
    {
      std::ostringstream oss;
      oss << "missing adjacent transform T[" << p << " -> " << (p - 1)
          << "] for getProjectionParam(" << curPoc << ", " << refPoc << ")";
      setError(oss.str());
      return false;
    }

    std::array<double, 9> RInv;
    std::array<double, 3> tInv;
    inverseRigid(rec->R, rec->t, RInv, tInv);
    composeLeft(RInv, tInv, out.R, out.t);
  }

  return true;
}

bool DepthCamParamBase::mapCurPixelToRef(
    const DepthCamProjectionParam& proj,
    double curX,
    double curY,
    double depthLinear,
    double& refX,
    double& refY)
{
  refX = -1.0;
  refY = -1.0;

  if (!(depthLinear > 0.0) || !std::isfinite(depthLinear))
  {
    return false;
  }

  const DepthCamIntrinsic& intr = proj.intr;
  if (std::fabs(intr.fx) < kEps || std::fabs(intr.fy) < kEps)
  {
    return false;
  }

  const double z = depthLinear;
  const std::array<double, 3> Pcur{{
      (curX - intr.cx) / intr.fx * z,
      (curY - intr.cy) / intr.fy * z,
      intr.zSign * z
  }};

  std::array<double, 3> RP;
  matVecMul3(proj.R, Pcur, RP);

  const double Xr = RP[0] + proj.t[0];
  const double Yr = RP[1] + proj.t[1];
  const double Zr = RP[2] + proj.t[2];

  if (!(Zr * intr.zSign > 0.0))
  {
    return false;
  }

  const double zDenom = std::fabs(Zr);
  if (zDenom < kEps)
  {
    return false;
  }

  refX = intr.fx * (Xr / zDenom) + intr.cx;
  refY = intr.fy * (Yr / zDenom) + intr.cy;

  return std::isfinite(refX) && std::isfinite(refY);
}

void DepthCamParamBase::setIdentity(std::array<double, 9>& R)
{
  R = {{1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0}};
}

void DepthCamParamBase::setZero(std::array<double, 3>& t)
{
  t = {{0.0, 0.0, 0.0}};
}

void DepthCamParamBase::rodriguesToMatrix(const double rvec[3], std::array<double, 9>& R)
{
  const double rx = rvec[0];
  const double ry = rvec[1];
  const double rz = rvec[2];
  const double theta = std::sqrt(rx * rx + ry * ry + rz * rz);

  if (theta < 1e-12)
  {
    // First-order approximation: R = I + [r]x.
    R = {{1.0, -rz,   ry,
          rz,   1.0, -rx,
         -ry,   rx,   1.0}};
    return;
  }

  const double kx = rx / theta;
  const double ky = ry / theta;
  const double kz = rz / theta;

  const double c = std::cos(theta);
  const double s = std::sin(theta);
  const double v = 1.0 - c;

  R[0] = c + kx * kx * v;
  R[1] = kx * ky * v - kz * s;
  R[2] = kx * kz * v + ky * s;

  R[3] = ky * kx * v + kz * s;
  R[4] = c + ky * ky * v;
  R[5] = ky * kz * v - kx * s;

  R[6] = kz * kx * v - ky * s;
  R[7] = kz * ky * v + kx * s;
  R[8] = c + kz * kz * v;
}

void DepthCamParamBase::matMul3x3(
    const std::array<double, 9>& A,
    const std::array<double, 9>& B,
    std::array<double, 9>& C)
{
  std::array<double, 9> T;
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      T[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c]
                   + A[r * 3 + 1] * B[1 * 3 + c]
                   + A[r * 3 + 2] * B[2 * 3 + c];
    }
  }
  C = T;
}

void DepthCamParamBase::matVecMul3(
    const std::array<double, 9>& A,
    const std::array<double, 3>& x,
    std::array<double, 3>& y)
{
  std::array<double, 3> t;
  t[0] = A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
  t[1] = A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
  t[2] = A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
  y = t;
}

void DepthCamParamBase::composeLeft(
    const std::array<double, 9>& RLeft,
    const std::array<double, 3>& tLeft,
    std::array<double, 9>& RAcc,
    std::array<double, 3>& tAcc)
{
  // New accumulated transform = TLeft * TAcc.
  // R = RLeft * RAcc
  // t = RLeft * tAcc + tLeft
  std::array<double, 9> RNew;
  matMul3x3(RLeft, RAcc, RNew);

  std::array<double, 3> Rt;
  matVecMul3(RLeft, tAcc, Rt);

  std::array<double, 3> tNew{{
      Rt[0] + tLeft[0],
      Rt[1] + tLeft[1],
      Rt[2] + tLeft[2]
  }};

  RAcc = RNew;
  tAcc = tNew;
}

void DepthCamParamBase::inverseRigid(
    const std::array<double, 9>& R,
    const std::array<double, 3>& t,
    std::array<double, 9>& RInv,
    std::array<double, 3>& tInv)
{
  // RInv = R^T.
  RInv[0] = R[0]; RInv[1] = R[3]; RInv[2] = R[6];
  RInv[3] = R[1]; RInv[4] = R[4]; RInv[5] = R[7];
  RInv[6] = R[2]; RInv[7] = R[5]; RInv[8] = R[8];

  std::array<double, 3> Rt;
  matVecMul3(RInv, t, Rt);
  tInv[0] = -Rt[0];
  tInv[1] = -Rt[1];
  tInv[2] = -Rt[2];
}

// ============================================================================
// Encoder JSONL parser
// ============================================================================

DepthCamParamEncoder::DepthCamParamEncoder(
    const std::string& jsonlPath,
    int width,
    int height,
    const DepthCamParamConfig& cfg)
  : DepthCamParamBase(width, height, cfg), m_jsonlPath(jsonlPath)
{
  m_fin.open(m_jsonlPath.c_str(), std::ios::in);
  if (!m_fin.is_open())
  {
    setError("failed to open jsonl: " + m_jsonlPath);
    m_valid = false;
    return;
  }

  if (!readHeaderFromJsonl())
  {
    m_valid = false;
    return;
  }
}

DepthCamParamEncoder::~DepthCamParamEncoder()
{
  if (m_fin.is_open())
  {
    m_fin.close();
  }
}

bool DepthCamParamEncoder::load(int poc, DepthCamQResidual& outQ)
{
  if (!isValid())
  {
    return false;
  }
  if (poc < 0)
  {
    setError("negative POC is not allowed");
    return false;
  }

  if (poc == 0)
  {
    DepthCamQResidual q0;
    q0.q = {{0, 0, 0, 0, 0, 0}};
    q0.clipped = false;
    if (!setAndStore(0, q0))
    {
      return false;
    }
    outQ = q0;
    return true;
  }

  JsonFrame jf;
  while (readNextFrame(jf))
  {
    if (jf.poc < poc)
    {
      // Allows optional poc=0 or skipped old frame lines in JSONL.
      continue;
    }

    if (jf.poc > poc)
    {
      std::ostringstream oss;
      oss << "jsonl frame POC jumped beyond requested POC. requested=" << poc
          << ", found=" << jf.poc;
      setError(oss.str());
      return false;
    }

    DepthCamParam6 gt;
    gt.v[0] = jf.rvec[0];
    gt.v[1] = jf.rvec[1];
    gt.v[2] = jf.rvec[2];
    gt.v[3] = jf.tvec[0] / getHeader().depthScale;
    gt.v[4] = jf.tvec[1] / getHeader().depthScale;
    gt.v[5] = jf.tvec[2] / getHeader().depthScale;

    return encodeAndStore(poc, gt, outQ);
  }

  std::ostringstream oss;
  oss << "failed to find POC " << poc << " in jsonl";
  setError(oss.str());
  return false;
}

bool DepthCamParamEncoder::readHeaderFromJsonl()
{
  std::string line;
  while (std::getline(m_fin, line))
  {
    if (isBlankLine(line))
    {
      continue;
    }

    double depthScale = 1.0;
    DepthCamIntrinsic intr;
    if (parseHeaderLine(line, depthScale, intr))
    {
      return initHeaderFromIntrinsic(depthScale, intr);
    }
  }

  setError("header line not found in jsonl");
  return false;
}

bool DepthCamParamEncoder::readNextFrame(JsonFrame& frame)
{
  std::string line;
  while (std::getline(m_fin, line))
  {
    if (isBlankLine(line))
    {
      continue;
    }

    JsonFrame tmp;
    if (parseFrameLine(line, tmp))
    {
      frame = tmp;
      return true;
    }
  }
  return false;
}

bool DepthCamParamEncoder::parseHeaderLine(const std::string& line, double& depthScale, DepthCamIntrinsic& intr)
{
  std::string type;
  if (!parseStringAfterKey(line, "type", type))
  {
    return false;
  }

  if (type != "header" && type != "intrinsic")
  {
    return false;
  }

  if (!parseDoubleAfterKey(line, "depth_scale", depthScale))
  {
    return false;
  }
  if (!parseDoubleAfterKey(line, "fx", intr.fx))
  {
    return false;
  }
  if (!parseDoubleAfterKey(line, "fy", intr.fy))
  {
    return false;
  }
  if (!parseDoubleAfterKey(line, "cx", intr.cx))
  {
    return false;
  }
  if (!parseDoubleAfterKey(line, "cy", intr.cy))
  {
    return false;
  }

  // Optional. Same default as the Python prototype.
  double zSign = -1.0;
  if (parseDoubleAfterKey(line, "z_sign", zSign))
  {
    intr.zSign = (zSign >= 0.0) ? 1.0 : -1.0;
  }
  else
  {
    intr.zSign = -1.0;
  }

  return true;
}

bool DepthCamParamEncoder::parseFrameLine(const std::string& line, JsonFrame& frame)
{
  if (!parseIntAfterKey(line, "poc", frame.poc))
  {
    return false;
  }
  if (!parseArray3AfterKey(line, "rvec", frame.rvec))
  {
    return false;
  }
  if (!parseArray3AfterKey(line, "tvec", frame.tvec))
  {
    return false;
  }
  return true;
}

// ============================================================================
// Decoder
// ============================================================================

DepthCamParamDecoder::DepthCamParamDecoder(
    const DepthCamHeader& headerFromBitstream,
    int width,
    int height,
    const DepthCamParamConfig& cfg)
  : DepthCamParamBase(width, height, cfg)
{
  if (!initHeaderFromQuant(headerFromBitstream))
  {
    m_valid = false;
  }
}

bool DepthCamParamDecoder::set(int poc, const DepthCamQResidual& q)
{
  if (poc < 0)
  {
    setError("negative POC is not allowed");
    return false;
  }
  return setAndStore(poc, q);
}

} // namespace depthcam


















void DepthCamParamBase::storeQResidual(int poc, const DepthCamQResidual& q)
{
  m_qResidualStore[poc] = q;
}

void DepthCamParamBase::clearQResidualStore()
{
  m_qResidualStore.clear();
}

bool DepthCamParamBase::hasQResidual(int poc) const
{
  return m_qResidualStore.find(poc) != m_qResidualStore.end();
}

bool DepthCamParamBase::getQResidual(int poc, DepthCamQResidual& out) const
{
  auto it = m_qResidualStore.find(poc);
  if (it == m_qResidualStore.end())
  {
    return false;
  }

  out = it->second;
  return true;
}

const DepthCamQResidual& DepthCamParamBase::getQResidualRef(int poc) const
{
  auto it = m_qResidualStore.find(poc);
  if (it == m_qResidualStore.end())
  {
    throw std::runtime_error("DepthCam q residual not found");
  }

  return it->second;
}















bool DepthCamParamDecoder::set(int poc, const DepthCamQResidual& q)
{
  // 1. q residual 먼저 저장
  storeQResidual(poc, q);

  // 2. predictor 생성
  DepthCamParam6 pred = predictFromHistory();

  // 3. q inverse scale
  DepthCamParam6 decResidual;
  for (int i = 0; i < 3; ++i)
  {
    decResidual.v[i] = q.q[i] * m_cfg.rStep;
  }
  for (int i = 3; i < 6; ++i)
  {
    decResidual.v[i] = q.q[i] * m_cfg.tStepNorm;
  }

  // 4. decoded param 생성
  DepthCamParam6 dec;
  for (int i = 0; i < 6; ++i)
  {
    dec.v[i] = pred.v[i] + decResidual.v[i];
  }

  // 5. history/store 등록
  storeDecodedParam(poc, dec);

  return true;
}











