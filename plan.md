#if GlobalMotion
#include <fstream>
#include <iomanip>
#include <string>
#endif

#if GlobalMotion
static void dumpMotionInfoCsv(const CodingStructure& cs, const std::string& fileName, const int poc)
{
  const Area lumaArea = cs.area.Y();

  const CMotionBuf mb = cs.getMotionBuf(lumaArea);

  const int miBlkW = 1 << g_miScaling.posx;   // normally 4
  const int miBlkH = 1 << g_miScaling.posy;   // normally 4

  const double mvScale = double(1 << MV_FRACTIONAL_BITS_INTERNAL);

  const bool writeHeader = !std::ifstream(fileName).good();

  std::ofstream ofs(fileName, std::ios::out | std::ios::app);
  CHECK(!ofs.good(), "Failed to open motion-info dump CSV");

  if (writeHeader)
  {
    ofs << "poc,x,y,w,h,isInter,interDir,list,refIdx,"
        << "mv_x_int,mv_y_int,mv_x_pel,mv_y_pel\n";
  }

  for (int my = 0; my < mb.height; my++)
  {
    for (int mx = 0; mx < mb.width; mx++)
    {
      const MotionInfo& mi = mb.at(mx, my);

      const int x = lumaArea.x + (mx << g_miScaling.posx);
      const int y = lumaArea.y + (my << g_miScaling.posy);

      const int w = std::min(miBlkW, lumaArea.x + lumaArea.width  - x);
      const int h = std::min(miBlkH, lumaArea.y + lumaArea.height - y);

      if (!mi.isInter)
      {
        ofs << poc << ","
            << x << "," << y << "," << w << "," << h << ","
            << 0 << ","
            << int(mi.interDir) << ","
            << "NA" << ","
            << -1 << ","
            << 0 << "," << 0 << ","
            << std::fixed << std::setprecision(6)
            << 0.0 << "," << 0.0 << "\n";
        continue;
      }

      if (mi.interDir & 1)
      {
        const Mv& mv = mi.mv[0];

        ofs << poc << ","
            << x << "," << y << "," << w << "," << h << ","
            << 1 << ","
            << int(mi.interDir) << ","
            << "L0" << ","
            << int(mi.refIdx[0]) << ","
            << mv.getHor() << ","
            << mv.getVer() << ","
            << std::fixed << std::setprecision(6)
            << double(mv.getHor()) / mvScale << ","
            << double(mv.getVer()) / mvScale << "\n";
      }

      if (mi.interDir & 2)
      {
        const Mv& mv = mi.mv[1];

        ofs << poc << ","
            << x << "," << y << "," << w << "," << h << ","
            << 1 << ","
            << int(mi.interDir) << ","
            << "L1" << ","
            << int(mi.refIdx[1]) << ","
            << mv.getHor() << ","
            << mv.getVer() << ","
            << std::fixed << std::setprecision(6)
            << double(mv.getHor()) / mvScale << ","
            << double(mv.getVer()) / mvScale << "\n";
      }
    }
  }
}
#endif
