# pipeline/ — P6: aşama soyutlaması + runner (arayüz/iskelet).
# Tüm modül arayüzlerini bilir (kareden track'e tüm zinciri kurar).
add_library(dtrack_pipeline INTERFACE)
target_include_directories(dtrack_pipeline INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dtrack_pipeline INTERFACE
  dtrack_core
  dtrack_io
  dtrack_stabilization
  dtrack_detection
  dtrack_tracking)
