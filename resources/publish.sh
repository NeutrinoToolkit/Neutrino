#!/bin/bash
set -e
if [[ "$GITHUB_TOKEN" == "" ]]; then
  echo "GitHub access token GITHUB_TOKEN env is not provided!"
  exit 1
fi
RETRIES=0
until [ $RETRIES -eq 20 ]
do
  echo "Finding the GitHub release associated with '$CIRRUS_TAG' tag..."
  CIRRUS_RELEASE=$(curl -sL -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json" https://api.github.com/repos/$CIRRUS_REPO_FULL_NAME/releases | jq -c "[ .[] | select( .tag_name | contains(\"$CIRRUS_TAG\")) ] | .[0]" | jq -r '.id')
  [[ "$CIRRUS_RELEASE" != "null" ]] && break
  RETRIES=$((RETRIES+1))
  sleep 30
done
if [[ "$CIRRUS_RELEASE" == "null" ]]; then
    echo "Can not find the associated GitHub '$CIRRUS_TAG' release!"
    exit 1
fi
echo "GitHub release '$CIRRUS_TAG' found. Preparing asset files to upload..."
file_content_type="application/octet-stream"

for fpath in $@
do
  echo "Uploading GitHub release asset '$fpath'..."
  name=$(basename "$fpath")
  url_to_upload="https://uploads.github.com/repos/$CIRRUS_REPO_FULL_NAME/releases/$CIRRUS_RELEASE/assets?name=$name"
  curl -LX POST \
    --data-binary @$fpath \
    --header "Authorization: token $GITHUB_TOKEN" \
    --header "Content-Type: $file_content_type" \
    $url_to_upload
done
echo
echo "GitHub release '$CIRRUS_TAG' assets uploaded successfully."
