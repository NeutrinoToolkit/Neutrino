# This cmake module sets the project version and partial version
# variables by analysing the git tag and commit history.

find_package(Git REQUIRED)

if (GIT_FOUND)
    # Get last tag from git
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=0 --tags
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    #How many commits since last tag
	execute_process(COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_AHEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE)

	# Get current commit SHA from git
	execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_GIT_SHA
		OUTPUT_STRIP_TRAILING_WHITESPACE)

else(GIT_FOUND)
    message( FATAL_ERROR "You need to install GIT")

endif(GIT_FOUND)


# Set project version (without the preceding 'v')
set(${PROJECT_NAME}_VERSION ${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH})
if (${PROJECT_NAME}_VERSION_TWEAK)
    set(${PROJECT_NAME}_VERSION ${${PROJECT_NAME}_VERSION}-${${PROJECT_NAME}_VERSION_TWEAK})
endif()



