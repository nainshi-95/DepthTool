#include "CameraParam.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

CameraMatrix4x4::CameraMatrix4x4()
{
  setIdentity();
}

void CameraMatrix4x4::setIdentity()
{
  m.fill(0.0);
  for (int i = 0; i < 4; i++)
  {
    (*this)(i, i) = 1.0;
  }
}

CameraRelativeParam::CameraRelativeParam()
{
  r.fill(0.0);
  t.fill(0.0);
  r[0] = r[4] = r[8] = 1.0;
}

CameraParamManager::CameraParamManager(int maxStoredParams)
  : m_maxStoredParams(std::max(1, maxStoredParams))
{
}

void CameraParamManager::clear()
{
  m_jsonFrameTextByPoc.clear();
  m_frameParamByPoc.clear();
  m_lruPocs.clear();
  m_jsonText.clear();
  m_jsonIndexed = false;
}

void CameraParamManager::setMaxStoredParams(int maxStoredParams)
{
  m_maxStoredParams = std::max(1, maxStoredParams);
  while ((int)m_lruPocs.size() > m_maxStoredParams)
  {
    const int oldPoc = m_lruPocs.front();
    m_lruPocs.pop_front();
    m_frameParamByPoc.erase(oldPoc);
  }
}

void CameraParamManager::setFrameSize(int width, int height)
{
  m_width  = width;
  m_height = height;
}

void CameraParamManager::setJsonFileName(const std::string &jsonFileName)
{
  m_jsonFileName = jsonFileName;
  m_jsonText.clear();
  m_jsonIndexed = false;
  m_jsonFrameTextByPoc.clear();
}

void CameraParamManager::loadJsonFile(const std::string &jsonFileName)
{
  setJsonFileName(jsonFileName);

  std::ifstream ifs(jsonFileName.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.good())
  {
    THROW("CameraParamManager: cannot open camera json file: " << jsonFileName);
  }

  std::ostringstream oss;
  oss << ifs.rdbuf();
  m_jsonText = oss.str();
  if (m_jsonText.empty())
  {
    THROW("CameraParamManager: empty camera json file: " << jsonFileName);
  }

  m_jsonIndexed = false;
  indexJsonIfNeeded();
}

void CameraParamManager::setFrameParam(const CameraFrameParam &param)
{
  if (!param.valid)
  {
    THROW("CameraParamManager: trying to set invalid camera parameter for POC " << param.poc);
  }
  storeFrameParam(param);
}

void CameraParamManager::setFrameParams(const std::vector<CameraFrameParam> &params)
{
  for (const CameraFrameParam &param : params)
  {
    setFrameParam(param);
  }
}

bool CameraParamManager::hasFrameParam(int poc) const
{
  return m_frameParamByPoc.find(poc) != m_frameParamByPoc.end();
}

const CameraFrameParam &CameraParamManager::loadFrameParam(int poc)
{
  auto it = m_frameParamByPoc.find(poc);
  if (it != m_frameParamByPoc.end())
  {
    touchLru(poc);
    return it->second;
  }

  if (!m_jsonFileName.empty())
  {
    indexJsonIfNeeded();
    auto jt = m_jsonFrameTextByPoc.find(poc);
    if (jt == m_jsonFrameTextByPoc.end())
    {
      THROW("CameraParamManager: camera parameter for POC " << poc << " is not found in json");
    }

    CameraFrameParam param = parseFrameParamFromJsonText(poc, jt->second);
    storeFrameParam(param);

    auto kt = m_frameParamByPoc.find(poc);
    if (kt == m_frameParamByPoc.end())
    {
      THROW("CameraParamManager: internal error after loading POC " << poc);
    }
    return kt->second;
  }

  THROW("CameraParamManager: camera parameter for POC " << poc << " is not available");
}

CameraRelativeParam CameraParamManager::loadCameraParam(int poc)
{
  if (poc == 0)
  {
    const CameraFrameParam &cur = loadFrameParam(0);
    CameraRelativeParam rel;
    rel.curPoc     = 0;
    rel.refPoc     = 0;
    rel.intrinsic  = cur.intrinsic;
    rel.depthScale = cur.depthScale;
    rel.valid      = true;
    return rel;
  }
  return loadRelativeParam(poc, poc - 1);
}

CameraRelativeParam CameraParamManager::loadRelativeParam(int curPoc, int refPoc)
{
  const CameraFrameParam &cur = loadFrameParam(curPoc);
  const CameraFrameParam &ref = loadFrameParam(refPoc);
  return deriveRelative(cur, ref);
}

void CameraParamManager::storeFrameParam(const CameraFrameParam &param)
{
  m_frameParamByPoc[param.poc] = param;
  touchLru(param.poc);

  while ((int)m_lruPocs.size() > m_maxStoredParams)
  {
    const int oldPoc = m_lruPocs.front();
    m_lruPocs.pop_front();
    if (oldPoc != param.poc)
    {
      m_frameParamByPoc.erase(oldPoc);
    }
  }
}

void CameraParamManager::touchLru(int poc)
{
  auto it = std::find(m_lruPocs.begin(), m_lruPocs.end(), poc);
  if (it != m_lruPocs.end())
  {
    m_lruPocs.erase(it);
  }
  m_lruPocs.push_back(poc);
}

void CameraParamManager::indexJsonIfNeeded()
{
  if (m_jsonIndexed)
  {
    return;
  }

  if (m_jsonText.empty())
  {
    if (m_jsonFileName.empty())
    {
      THROW("CameraParamManager: json file name is not set");
    }

    std::ifstream ifs(m_jsonFileName.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.good())
    {
      THROW("CameraParamManager: cannot open camera json file: " << m_jsonFileName);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    m_jsonText = oss.str();
  }

  bool foundDepthScale = false;
  m_depthScale = parseJsonNumber(m_jsonText, { "depthScale", "DepthScale", "depth_scale", "DepthScaleFactor" }, m_depthScale, &foundDepthScale);

  std::string framesArray;
  std::vector<std::string> entries;

  if (findJsonValue(m_jsonText, { "frames" }, framesArray) && !framesArray.empty() && framesArray[0] == '[')
  {
    entries = splitTopLevelObjectsFromArray(framesArray);
  }
  else
  {
    const std::string root = trim(m_jsonText);
    if (!root.empty() && root[0] == '[')
    {
      entries = splitTopLevelObjectsFromArray(root);
    }
    else if (hasCameraMatrices(root))
    {
      entries.push_back(root);
    }
    else
    {
      for (const auto &kv : splitTopLevelMembers(root))
      {
        const std::string value = trim(kv.second);
        if (!value.empty() && value[0] == '{' && hasCameraMatrices(value))
        {
          entries.push_back(value);
        }
      }
    }
  }

  if (entries.empty())
  {
    THROW("CameraParamManager: no camera frame entries found in json: " << m_jsonFileName);
  }

  for (int i = 0; i < (int)entries.size(); i++)
  {
    const int poc = parsePocFromFrameText(entries[i], i);
    m_jsonFrameTextByPoc[poc] = entries[i];
    if (m_jsonFrameTextByPoc.find(i) == m_jsonFrameTextByPoc.end())
    {
      m_jsonFrameTextByPoc[i] = entries[i];
    }
  }

  m_jsonIndexed = true;
}

CameraFrameParam CameraParamManager::parseFrameParamFromJsonText(int poc, const std::string &frameText) const
{
  CameraFrameParam param;
  param.poc = poc;

  param.depthScale    = m_depthScale;
  param.nearClipPlane = parseJsonNumber(frameText, { "nearClipPlane", "NearClipPlane" }, 1.0);

  param.invProjectionMatrix = parseMatrix(frameText, { "InvProjectionMatrix", "invProjectionMatrix" }, true);

  bool hasProj = false;
  std::string projText;
  hasProj = findJsonValue(frameText, { "ProjectionMatrix", "projectionMatrix" }, projText);
  if (hasProj)
  {
    param.projectionMatrix = parseMatrix(frameText, { "ProjectionMatrix", "projectionMatrix" }, true);
  }
  else
  {
    param.projectionMatrix.setIdentity();
  }

  param.worldToCameraMatrix = parseMatrix(frameText, { "WorldToCameraMatrix", "worldToCameraMatrix" }, true);
  param.cameraToWorldMatrix = parseMatrix(frameText,
                                          { "CameraToWorldMatrix", "cameraToWorldMatrix", "CameraToWorldMarix", "cameraToWorldMarix" },
                                          true);

  if (m_width > 0 && m_height > 0)
  {
    param.intrinsic = deriveIntrinsic(param.invProjectionMatrix, m_width, m_height);
  }
  else
  {
    param.intrinsic.valid = false;
  }

  param.valid = true;
  return param;
}

CameraMatrix4x4 CameraParamManager::multiply4x4(const CameraMatrix4x4 &a, const CameraMatrix4x4 &b)
{
  CameraMatrix4x4 c;
  c.m.fill(0.0);
  for (int r = 0; r < 4; r++)
  {
    for (int col = 0; col < 4; col++)
    {
      double sum = 0.0;
      for (int k = 0; k < 4; k++)
      {
        sum += a(r, k) * b(k, col);
      }
      c(r, col) = sum;
    }
  }
  return c;
}

CameraMatrix4x4 CameraParamManager::transpose4x4(const CameraMatrix4x4 &a)
{
  CameraMatrix4x4 t;
  for (int r = 0; r < 4; r++)
  {
    for (int c = 0; c < 4; c++)
    {
      t(r, c) = a(c, r);
    }
  }
  return t;
}

CameraIntrinsicParam CameraParamManager::deriveIntrinsic(const CameraMatrix4x4 &invProjectionMatrix, int width, int height)
{
  CameraIntrinsicParam intr;

  if (width <= 0 || height <= 0)
  {
    return intr;
  }

  constexpr int numX = 32;
  constexpr int numY = 18;

  double sumRx = 0.0, sumRx2 = 0.0, sumU = 0.0, sumRxU = 0.0;
  double sumRy = 0.0, sumRy2 = 0.0, sumV = 0.0, sumRyV = 0.0;
  int    n = 0;
  std::vector<double> qzList;
  qzList.reserve(numX * numY);

  for (int iy = 0; iy < numY; iy++)
  {
    const double v = (numY == 1) ? 0.0 : (double)iy * (double)(height - 1) / (double)(numY - 1);
    for (int ix = 0; ix < numX; ix++)
    {
      const double u = (numX == 1) ? 0.0 : (double)ix * (double)(width - 1) / (double)(numX - 1);

      const double xNdc = (u + 0.5) / (double)width * 2.0 - 1.0;
      const double yNdc = 1.0 - (v + 0.5) / (double)height * 2.0;

      const double p[4] = { xNdc, yNdc, 1.0, 1.0 };
      double q[4] = { 0.0, 0.0, 0.0, 0.0 };

      for (int r = 0; r < 4; r++)
      {
        for (int c = 0; c < 4; c++)
        {
          q[r] += invProjectionMatrix(r, c) * p[c];
        }
      }

      const double qw = std::max(std::abs(q[3]), 1e-8);
      q[0] /= qw;
      q[1] /= qw;
      q[2] /= qw;

      const double zAbs = std::max(std::abs(q[2]), 1e-8);
      const double rx = q[0] / zAbs;
      const double ry = q[1] / zAbs;

      sumRx  += rx;
      sumRx2 += rx * rx;
      sumU   += u;
      sumRxU += rx * u;

      sumRy  += ry;
      sumRy2 += ry * ry;
      sumV   += v;
      sumRyV += ry * v;

      qzList.push_back(q[2]);
      n++;
    }
  }

  const double denX = (double)n * sumRx2 - sumRx * sumRx;
  const double denY = (double)n * sumRy2 - sumRy * sumRy;

  if (std::abs(denX) < 1e-12 || std::abs(denY) < 1e-12)
  {
    return intr;
  }

  intr.fx = ((double)n * sumRxU - sumRx * sumU) / denX;
  intr.cx = (sumU - intr.fx * sumRx) / (double)n;

  intr.fy = ((double)n * sumRyV - sumRy * sumV) / denY;
  intr.cy = (sumV - intr.fy * sumRy) / (double)n;

  std::sort(qzList.begin(), qzList.end());
  const double medianZ = qzList[qzList.size() / 2];
  intr.zSign = (medianZ >= 0.0) ? 1.0 : -1.0;
  if (medianZ == 0.0)
  {
    intr.zSign = -1.0;
  }

  intr.valid = true;
  return intr;
}

CameraRelativeParam CameraParamManager::deriveRelative(const CameraFrameParam &cur, const CameraFrameParam &ref)
{
  CameraRelativeParam rel;
  rel.curPoc = cur.poc;
  rel.refPoc = ref.poc;

  // X_ref = W2C_ref * C2W_cur * X_cur
  const CameraMatrix4x4 tCurToRef = multiply4x4(ref.worldToCameraMatrix, cur.cameraToWorldMatrix);

  for (int r = 0; r < 3; r++)
  {
    for (int c = 0; c < 3; c++)
    {
      rel.r[r * 3 + c] = tCurToRef(r, c);
    }
    rel.t[r] = tCurToRef(r, 3);
  }

  rel.intrinsic  = cur.intrinsic;
  rel.depthScale = cur.depthScale;
  rel.valid      = cur.valid && ref.valid && cur.intrinsic.valid;
  return rel;
}

bool CameraParamManager::hasCameraMatrices(const std::string &text)
{
  std::string dummy;
  return findJsonValue(text, { "InvProjectionMatrix", "invProjectionMatrix" }, dummy)
      && findJsonValue(text, { "WorldToCameraMatrix", "worldToCameraMatrix" }, dummy)
      && findJsonValue(text, { "CameraToWorldMatrix", "cameraToWorldMatrix", "CameraToWorldMarix", "cameraToWorldMarix" }, dummy);
}

int CameraParamManager::parsePocFromFrameText(const std::string &frameText, int defaultPoc)
{
  bool found = false;
  const double poc = parseJsonNumber(frameText, { "frames", "frame", "frameIdx", "frame_idx", "poc", "POC" }, (double)defaultPoc, &found);
  return found ? (int)std::llround(poc) : defaultPoc;
}

bool CameraParamManager::findJsonValue(const std::string &text, const std::vector<std::string> &keys, std::string &valueText)
{
  for (const std::string &key : keys)
  {
    const std::string quotedKey = std::string("\"") + key + "\"";
    const size_t keyPos = text.find(quotedKey);
    if (keyPos == std::string::npos)
    {
      continue;
    }

    size_t colon = text.find(':', keyPos + quotedKey.size());
    if (colon == std::string::npos)
    {
      continue;
    }

    size_t pos = colon + 1;
    while (pos < text.size() && std::isspace((unsigned char)text[pos]))
    {
      pos++;
    }
    if (pos >= text.size())
    {
      continue;
    }

    size_t end = pos;
    if (text[pos] == '{' || text[pos] == '[')
    {
      const char openCh = text[pos];
      const char closeCh = (openCh == '{') ? '}' : ']';
      int depth = 0;
      bool inString = false;
      bool escape = false;
      for (; end < text.size(); end++)
      {
        const char ch = text[end];
        if (inString)
        {
          if (escape)
          {
            escape = false;
          }
          else if (ch == '\\')
          {
            escape = true;
          }
          else if (ch == '"')
          {
            inString = false;
          }
          continue;
        }
        if (ch == '"')
        {
          inString = true;
          continue;
        }
        if (ch == openCh)
        {
          depth++;
        }
        else if (ch == closeCh)
        {
          depth--;
          if (depth == 0)
          {
            end++;
            break;
          }
        }
      }
    }
    else if (text[pos] == '"')
    {
      end = pos + 1;
      bool escape = false;
      for (; end < text.size(); end++)
      {
        const char ch = text[end];
        if (escape)
        {
          escape = false;
        }
        else if (ch == '\\')
        {
          escape = true;
        }
        else if (ch == '"')
        {
          end++;
          break;
        }
      }
    }
    else
    {
      while (end < text.size() && text[end] != ',' && text[end] != '}' && text[end] != ']' && !std::isspace((unsigned char)text[end]))
      {
        end++;
      }
    }

    valueText = trim(text.substr(pos, end - pos));
    return true;
  }

  return false;
}

double CameraParamManager::parseJsonNumber(const std::string &text, const std::vector<std::string> &keys, double defaultValue, bool *found)
{
  std::string valueText;
  const bool ok = findJsonValue(text, keys, valueText);
  if (found)
  {
    *found = ok;
  }
  if (!ok)
  {
    return defaultValue;
  }

  const std::vector<double> numbers = extractNumbers(valueText);
  if (numbers.empty())
  {
    if (found)
    {
      *found = false;
    }
    return defaultValue;
  }
  return numbers[0];
}

CameraMatrix4x4 CameraParamManager::parseMatrix(const std::string &frameText, const std::vector<std::string> &keys, bool transposeObjectMatrix)
{
  std::string valueText;
  if (!findJsonValue(frameText, keys, valueText))
  {
    THROW("CameraParamManager: missing matrix in camera json");
  }

  CameraMatrix4x4 mat;
  mat.m.fill(0.0);

  const std::string value = trim(valueText);
  if (!value.empty() && value[0] == '{')
  {
    for (int r = 0; r < 4; r++)
    {
      for (int c = 0; c < 4; c++)
      {
        const std::string e = std::string("e") + char('0' + r) + char('0' + c);
        bool found = false;
        mat(r, c) = parseJsonNumber(value, { e }, 0.0, &found);
        if (!found)
        {
          THROW("CameraParamManager: missing matrix element " << e);
        }
      }
    }
    return transposeObjectMatrix ? transpose4x4(mat) : mat;
  }

  std::vector<double> nums = extractNumbers(value);
  if (nums.size() != 16)
  {
    THROW("CameraParamManager: matrix array must have 16 numbers, got " << nums.size());
  }

  for (int i = 0; i < 16; i++)
  {
    mat.m[i] = nums[i];
  }
  return mat;
}

std::vector<std::string> CameraParamManager::splitTopLevelObjectsFromArray(const std::string &arrayText)
{
  std::vector<std::string> out;
  const std::string s = trim(arrayText);
  if (s.empty())
  {
    return out;
  }

  bool inString = false;
  bool escape = false;
  int depth = 0;
  size_t objStart = std::string::npos;

  for (size_t i = 0; i < s.size(); i++)
  {
    const char ch = s[i];
    if (inString)
    {
      if (escape)
      {
        escape = false;
      }
      else if (ch == '\\')
      {
        escape = true;
      }
      else if (ch == '"')
      {
        inString = false;
      }
      continue;
    }

    if (ch == '"')
    {
      inString = true;
      continue;
    }
    if (ch == '{')
    {
      if (depth == 0)
      {
        objStart = i;
      }
      depth++;
    }
    else if (ch == '}')
    {
      depth--;
      if (depth == 0 && objStart != std::string::npos)
      {
        out.push_back(s.substr(objStart, i - objStart + 1));
        objStart = std::string::npos;
      }
    }
  }

  return out;
}

std::vector<std::pair<std::string, std::string>> CameraParamManager::splitTopLevelMembers(const std::string &objectText)
{
  std::vector<std::pair<std::string, std::string>> members;
  const std::string s = trim(objectText);
  if (s.size() < 2 || s.front() != '{' || s.back() != '}')
  {
    return members;
  }

  size_t pos = 1;
  while (pos + 1 < s.size())
  {
    while (pos < s.size() && (std::isspace((unsigned char)s[pos]) || s[pos] == ','))
    {
      pos++;
    }
    if (pos >= s.size() || s[pos] == '}')
    {
      break;
    }
    if (s[pos] != '"')
    {
      break;
    }

    const size_t keyStart = pos + 1;
    size_t keyEnd = keyStart;
    while (keyEnd < s.size() && s[keyEnd] != '"')
    {
      keyEnd++;
    }
    if (keyEnd >= s.size())
    {
      break;
    }
    const std::string key = s.substr(keyStart, keyEnd - keyStart);

    size_t colon = s.find(':', keyEnd + 1);
    if (colon == std::string::npos)
    {
      break;
    }
    pos = colon + 1;
    while (pos < s.size() && std::isspace((unsigned char)s[pos]))
    {
      pos++;
    }

    const size_t valueStart = pos;
    size_t valueEnd = valueStart;

    if (pos < s.size() && (s[pos] == '{' || s[pos] == '['))
    {
      const char openCh = s[pos];
      const char closeCh = (openCh == '{') ? '}' : ']';
      int depth = 0;
      bool inString = false;
      bool escape = false;
      for (; valueEnd < s.size(); valueEnd++)
      {
        const char ch = s[valueEnd];
        if (inString)
        {
          if (escape) escape = false;
          else if (ch == '\\') escape = true;
          else if (ch == '"') inString = false;
          continue;
        }
        if (ch == '"')
        {
          inString = true;
          continue;
        }
        if (ch == openCh) depth++;
        else if (ch == closeCh)
        {
          depth--;
          if (depth == 0)
          {
            valueEnd++;
            break;
          }
        }
      }
    }
    else
    {
      while (valueEnd < s.size() && s[valueEnd] != ',' && s[valueEnd] != '}')
      {
        valueEnd++;
      }
    }

    members.push_back({ key, trim(s.substr(valueStart, valueEnd - valueStart)) });
    pos = valueEnd;
  }

  return members;
}

std::vector<double> CameraParamManager::extractNumbers(const std::string &text)
{
  std::vector<double> nums;
  const char *begin = text.c_str();
  char *end = nullptr;

  for (size_t i = 0; i < text.size(); )
  {
    const char ch = text[i];
    const bool possibleStart = (ch == '-') || (ch == '+') || (ch == '.') || std::isdigit((unsigned char)ch);
    if (!possibleStart)
    {
      i++;
      continue;
    }

    const double v = std::strtod(begin + i, &end);
    if (end != begin + i)
    {
      nums.push_back(v);
      i = (size_t)(end - begin);
    }
    else
    {
      i++;
    }
  }
  return nums;
}

std::string CameraParamManager::trim(const std::string &s)
{
  size_t b = 0;
  while (b < s.size() && std::isspace((unsigned char)s[b]))
  {
    b++;
  }
  size_t e = s.size();
  while (e > b && std::isspace((unsigned char)s[e - 1]))
  {
    e--;
  }
  return s.substr(b, e - b);
}
