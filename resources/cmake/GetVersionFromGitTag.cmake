# This cmake module sets the project version and partial version
# variables by analysing the git tag and commit history.

find_package(Git REQUIRED)

if (GIT_FOUND)
    # Get branch
    if(DEFINED ENV{TRAVIS_BRANCH})
        set(${PROJECT_NAME}_VERSION_BRANCH $ENV{TRAVIS_BRANCH})
    else()
        execute_process(COMMAND ${GIT_EXECUTABLE} symbolic-ref HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE ${PROJECT_NAME}_PARTIAL_BRANCH
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        string(REPLACE "refs/heads/" "" ${PROJECT_NAME}_VERSION_BRANCH ${${PROJECT_NAME}_PARTIAL_BRANCH})
        unset(${PROJECT_NAME}_PARTIAL_BRANCH)
    endif()

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
	execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_GIT_SHA
		OUTPUT_STRIP_TRAILING_WHITESPACE)

	# Get current commit SHA from git for latest tag
	execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse latest
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_GIT_SHA_LATEST
		OUTPUT_STRIP_TRAILING_WHITESPACE)

	# Get partial versions into a list
    string(REGEX MATCHALL "-.*$|[0-9]+" ${PROJECT_NAME}_PARTIAL_VERSION_LIST
        ${${PROJECT_NAME}_VERSION_STRING})
    # The tweak part is optional, so check if the list contains it
    list(LENGTH ${PROJECT_NAME}_PARTIAL_VERSION_LIST
        ${PROJECT_NAME}_PARTIAL_VERSION_LIST_LEN)

    set(${PROJECT_NAME}_VERSION_PATCH ${${PROJECT_NAME}_VERSION_AHEAD})

    # Set the version numbers
    if (${PROJECT_NAME}_PARTIAL_VERSION_LIST_LEN GREATER 0)
        list(GET ${PROJECT_NAME}_PARTIAL_VERSION_LIST
            0 ${PROJECT_NAME}_VERSION_MAJOR)
        if (${PROJECT_NAME}_PARTIAL_VERSION_LIST_LEN GREATER 1)
            list(GET ${PROJECT_NAME}_PARTIAL_VERSION_LIST
                1 ${PROJECT_NAME}_VERSION_MINOR)
            if (${PROJECT_NAME}_PARTIAL_VERSION_LIST_LEN GREATER 2)
                list(GET ${PROJECT_NAME}_PARTIAL_VERSION_LIST
                    2 ${PROJECT_NAME}_VERSION_PATCH)
                if (${PROJECT_NAME}_PARTIAL_VERSION_LIST_LEN GREATER 3)

                    list(GET ${PROJECT_NAME}_PARTIAL_VERSION_LIST 3 ${PROJECT_NAME}_VERSION_TWEAK)
                    string(SUBSTRING ${${PROJECT_NAME}_VERSION_TWEAK} 1 -1 ${PROJECT_NAME}_VERSION_TWEAK)
                endif()
            endif()
        endif()
    endif()

    # Unset the list
    unset(${PROJECT_NAME}_PARTIAL_VERSION_LIST)

    # Set full project version string
    set(${PROJECT_NAME}_VERSION_STRING_FULL
        ${${PROJECT_NAME}_VERSION_BRANCH}-${${PROJECT_NAME}_VERSION_STRING}-${${PROJECT_NAME}_VERSION_AHEAD}.${${PROJECT_NAME}_VERSION_GIT_SHA})

else(GIT_FOUND)
    message( FATAL_ERROR "You need to install GIT")

endif(GIT_FOUND)


# Set project version (without the preceding 'v')
set(${PROJECT_NAME}_VERSION ${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH})
if (${PROJECT_NAME}_VERSION_TWEAK)
    set(${PROJECT_NAME}_VERSION ${${PROJECT_NAME}_VERSION}-${${PROJECT_NAME}_VERSION_TWEAK})
endif()



