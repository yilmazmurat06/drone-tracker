# dtrack_pipeline: Stage thread iskeleti + Pipeline orkestratör + IStage.
# Header-only -> INTERFACE. Thread'lere ve common'a ihtiyaç duyar.
add_library(dtrack_pipeline INTERFACE)

target_include_directories(dtrack_pipeline INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(dtrack_pipeline INTERFACE
    dtrack_common
    Threads::Threads)
