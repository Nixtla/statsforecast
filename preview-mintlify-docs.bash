#!/usr/bin/env bash

echo "Running nbdev_docs..."
nbdev_docs
echo "nbdev_docs is done"

echo "Running final-formatting.bash..."
chmod +x final-formatting.bash
./final-formatting.bash
echo "final-formatting.bash is done"

echo "Moving necessary assets..."
cp nbs/mint.json _docs/mint.json
cp nbs/imgs/logo/dark.png _docs/dark.png
cp nbs/imgs/logo/light.png _docs/light.png 
cp nbs/favicon.svg _docs/favicon.svg
echo "Done moving necessary assets"

cd ./_docs
mintlify dev