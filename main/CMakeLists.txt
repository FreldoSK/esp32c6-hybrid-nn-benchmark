idf_component_register(SRCS "main.cpp" "assets.S"
                    INCLUDE_DIRS ".")

# .incbin looks files up relative to this directory; rebuild when they change.
target_include_directories(${COMPONENT_LIB} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB ASSET_FILES "${CMAKE_CURRENT_SOURCE_DIR}/models/*.tflite" "${CMAKE_CURRENT_SOURCE_DIR}/test_*.bin")
set_property(SOURCE assets.S APPEND PROPERTY OBJECT_DEPENDS ${ASSET_FILES})
