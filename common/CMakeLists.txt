# dtrack_common: paylaşılan tipler + lock-free ring buffer.
# Header-only -> INTERFACE kütüphane (derlenecek .cpp yok; sadece include taşır).
add_library(dtrack_common INTERFACE)

target_include_directories(dtrack_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# types.hpp OpenCV'ye ihtiyaç duyar; varsa bağla. (ring_buffer.hpp / time.hpp duymaz,
# bu yüzden OpenCV olmadan da ring buffer testi bu hedefi kullanabilir.)
if(OpenCV_FOUND)
    target_link_libraries(dtrack_common INTERFACE ${OpenCV_LIBS})
    target_include_directories(dtrack_common INTERFACE ${OpenCV_INCLUDE_DIRS})
endif()
