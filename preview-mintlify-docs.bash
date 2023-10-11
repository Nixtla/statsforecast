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
mkdir _docs/docs/contribute
cp _proc/docs/contribute/contribute.md _docs/docs/contribute/contribute.mdx
cp _proc/docs/contribute/docs.md _docs/docs/contribute/docs.mdx
cp _proc/docs/contribute/issue-labels.md _docs/docs/contribute/issue-labels.mdx
cp _proc/docs/contribute/issues.md _docs/docs/contribute/issues.mdx
cp _proc/docs/contribute/step-by-step.md _docs/docs/contribute/step-by-step.mdx
cp _proc/docs/contribute/techstack.md _docs/docs/contribute/techstack.mdx
echo "Done moving necessary assets"

cd ./_docs
mintlify dev