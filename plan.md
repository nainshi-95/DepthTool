#if GlobalMotion
#include <fstream>
#include <iomanip>
#include <string>
#endif

#if GlobalMotion
static void dumpMotionInfoCsvPelOnly(
  const CodingStructure& cs,
  const std::string& fileName,
  const int currPoc
)
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
    ofs << "poc,x,y,w,h,list,ref_poc,mv_x,mv_y\n";
  }

  for (int my = 0; my < mb.height; my++)
  {
    for (int mx = 0; mx < mb.width; mx++)
    {
      const MotionInfo& mi = mb.at(mx, my);

      // No isInter column. Just skip non-inter blocks.
      if (!mi.isInter)
      {
        continue;
      }

      const int x = lumaArea.x + (mx << g_miScaling.posx);
      const int y = lumaArea.y + (my << g_miScaling.posy);

      const int w = std::min(miBlkW, lumaArea.x + lumaArea.width  - x);
      const int h = std::min(miBlkH, lumaArea.y + lumaArea.height - y);

      const Slice* slice = nullptr;

      if (cs.picture && mi.sliceIdx < cs.picture->m_slices.size())
      {
        slice = cs.picture->m_slices[mi.sliceIdx];
      }
      else
      {
        slice = cs.slice;
      }

      CHECK(slice == nullptr, "Cannot resolve slice for MotionInfo dump");

      auto writeOneList = [&](const RefPicList refList, const char* listName)
      {
        const int listIdx = int(refList);
        const int refIdx = mi.refIdx[listIdx];

        if (refIdx < 0)
        {
          return;
        }

        const Picture* refPic = slice->getRefPic(refList, refIdx);
        CHECK(refPic == nullptr, "Cannot resolve reference picture for MotionInfo dump");

        const int refPoc = refPic->m_poc;
        const Mv& mv = mi.mv[listIdx];

        const double mvXPel = double(mv.getHor()) / mvScale;
        const double mvYPel = double(mv.getVer()) / mvScale;

        ofs << currPoc << ","
            << x << "," << y << "," << w << "," << h << ","
            << listName << ","
            << refPoc << ","
            << std::fixed << std::setprecision(6)
            << mvXPel << ","
            << mvYPel << "\n";
      };

      if (mi.interDir & 1)
      {
        writeOneList(RPL0, "L0");
      }

      if (mi.interDir & 2)
      {
        writeOneList(RPL1, "L1");
      }
    }
  }
}
#endif
