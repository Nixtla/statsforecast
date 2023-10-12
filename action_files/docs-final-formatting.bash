#!/usr/bin/env bash

for file in $(find _docs -type f -name "*mdx"); do
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' -e 's/style="float:right; font-size:smaller"/style={{ float: "right", fontSize: "smaller" }}/g' $file
    sed -i '' -e 's/<br>/<br\/>/g' $file
  else
    sed -i -e 's/style="float:right; font-size:smaller"/style={{ float: "right", fontSize: "smaller" }}/g' $file
    sed -i -e 's/<br>/<br\/>/g' $file
  fi
done

python3 "$(dirname "$0")/docs_replace_imgs.py"