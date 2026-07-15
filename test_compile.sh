clang++ -O3 -shared -std=c++20 -fPIC \
  -I"pokemon-tcg-ai-battle/ptcg_engine/ptcgProgram 22" \
  "pokemon-tcg-ai-battle/ptcg_engine/ptcgProgram 22/Export.cpp" \
  "pokemon-tcg-ai-battle/ptcg_engine/ptcgProgram 22/RelationalObservation.cpp" \
  -o src/cg/libcg.dylib
