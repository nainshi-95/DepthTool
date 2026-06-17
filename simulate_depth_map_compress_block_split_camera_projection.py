1. depth plane model:
   Z = ax + by + c
   → invDepth = 1 / Z = ax + by + c

2. camera plane candidate:
   previous reconstructed inverse-depth plane
   → 3D plane fitting
   → camera transform
   → current block depth render
   → inverse-depth plane으로 다시 fitting

3. reconstructed plane depth로 backward projection:
   current reconstructed depth
   → current pixel back-project
   → previous camera로 project
   → previous GT YUV frame bilinear sampling
   → current GT frame과 PSNR 측정
   → predicted YUV420p10le 저장
