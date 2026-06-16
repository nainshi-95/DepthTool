/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 */

#ifndef __CAMERAPARAM__
#define __CAMERAPARAM__

#include "CommonDef.h"

#include <array>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

//! \ingroup CommonLib
//! \{

struct CameraMatrix4x4
{
  std::array<double, 16> m;

  CameraMatrix4x4();

  double       &operator()(int r, int c)       { return m[r * 4 + c]; }
  const double &operator()(int r, int c) const { return m[r * 4 + c]; }

  void setIdentity();
};

struct CameraIntrinsicParam
{
  double fx    = 0.0;
  double fy    = 0.0;
  double cx    = 0.0;
  double cy    = 0.0;
  double zSign = -1.0;
  bool   valid = false;
};

struct CameraFrameParam
{
  int poc = -1;

  CameraMatrix4x4 invProjectionMatrix;
  CameraMatrix4x4 projectionMatrix;
  CameraMatrix4x4 worldToCameraMatrix;
  CameraMatrix4x4 cameraToWorldMatrix;

  // Linear depth conversion:
  //   depthLinear = codedDepthValue * depthScale
  double depthScale = 10.0;

  CameraIntrinsicParam intrinsic;
  bool valid = false;
};

struct CameraRelativeParam
{
  int curPoc = -1;
  int refPoc = -1;

  // X_ref = R * X_cur + t
  std::array<double, 9> r;
  std::array<double, 3> t;

  CameraIntrinsicParam intrinsic;
  double depthScale = 10.0;
  bool valid = false;

  CameraRelativeParam();
};

class CameraParamManager
{
public:
  explicit CameraParamManager(int maxStoredParams = 10);

  void clear();
  void setMaxStoredParams(int maxStoredParams);
  int  getMaxStoredParams() const { return m_maxStoredParams; }

  // Encoder-side API.
  // The JSON file is indexed once, and each POC is converted lazily.
  void setJsonFileName(const std::string &jsonFileName);
  void loadJsonFile(const std::string &jsonFileName);
  bool hasJsonFile() const { return !m_jsonFileName.empty(); }

  // Decoder-side API.
  // Decoder must provide already decoded absolute camera parameters.
  void setFrameParam(const CameraFrameParam &param);
  void setFrameParams(const std::vector<CameraFrameParam> &params);

  // Load one absolute frame camera parameter.
  // Encoder can lazy-load from JSON.
  // Decoder errors if the requested POC does not exist.
  const CameraFrameParam &loadFrameParam(int poc);
  bool hasFrameParam(int poc) const;

  // General camera transform for inter prediction.
  // curPoc -> refPoc.
  CameraRelativeParam loadRelativeParam(int curPoc, int refPoc);

  double getDepthScale() const { return m_depthScale; }
  void   setDepthScale(double depthScale) { m_depthScale = depthScale; }

  void setFrameSize(int width, int height);
  int  getWidth()  const { return m_width; }
  int  getHeight() const { return m_height; }

private:
  struct JsonFrameEntry
  {
    int         poc = -1;
    std::string text;
  };

private:
  int m_maxStoredParams = 10;
  int m_width  = 0;
  int m_height = 0;

  double      m_depthScale = 10.0;
  std::string m_jsonFileName;
  std::string m_jsonText;
  bool        m_jsonIndexed = false;

  std::unordered_map<int, std::string>      m_jsonFrameTextByPoc;
  std::unordered_map<int, CameraFrameParam> m_frameParamByPoc;
  std::deque<int>                           m_lruPocs;

private:
  void indexJsonIfNeeded();
  CameraFrameParam parseFrameParamFromJsonText(int poc, const std::string &frameText) const;
  void storeFrameParam(const CameraFrameParam &param);
  void touchLru(int poc);

  static CameraMatrix4x4 multiply4x4(const CameraMatrix4x4 &a, const CameraMatrix4x4 &b);
  static CameraMatrix4x4 transpose4x4(const CameraMatrix4x4 &a);

  static CameraIntrinsicParam deriveIntrinsic(
    const CameraMatrix4x4 &invProjectionMatrix,
    int width,
    int height
  );

  static CameraRelativeParam deriveRelative(
    const CameraFrameParam &cur,
    const CameraFrameParam &ref
  );

  static bool hasCameraMatrices(const std::string &text);
  static int  parsePocFromFrameText(const std::string &frameText, int defaultPoc);

  static bool findJsonValue(
    const std::string &text,
    const std::vector<std::string> &keys,
    std::string &valueText
  );

  static double parseJsonNumber(
    const std::string &text,
    const std::vector<std::string> &keys,
    double defaultValue,
    bool *found = nullptr
  );

  static CameraMatrix4x4 parseMatrix(
    const std::string &frameText,
    const std::vector<std::string> &keys,
    bool transposeObjectMatrix
  );

  static std::vector<std::string> splitTopLevelObjectsFromArray(const std::string &arrayText);
  static std::vector<std::pair<std::string, std::string>> splitTopLevelMembers(const std::string &objectText);
  static std::vector<double> extractNumbers(const std::string &text);
  static std::string trim(const std::string &s);
};

//! \}

#endif // __CAMERAPARAM__
