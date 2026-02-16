add_library(utils STATIC io_utils.c logger.c)

target_link_libraries(utils PRIVATE matrix)
