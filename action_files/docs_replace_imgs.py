import os
from pathlib import Path
import re

def main():
  dir_path = os.path.dirname(os.path.realpath(__file__))
  docs_path = os.path.abspath(os.path.join(dir_path, '../_docs'))
  docs_path_obj = Path(docs_path)
  paths_in_dir = docs_path_obj.glob('**/*.mdx')
  pattern = r'!\[\]\(((?!https:\/\/).+?)\)'

  for file_path in paths_in_dir:
    if file_path.is_file():
      folder_arr = file_path.parts[len(docs_path_obj.parts):-1]
      content = file_path.read_text()
      content = re.sub(pattern, url_replace(folder_arr), content)
      file_path.write_text(content)

def url_replace(folder_arr):
  def url_replace_helper(match):
    curr_url = Path(match[1])
    new_url = str(Path(*folder_arr).joinpath(curr_url))
    if not new_url.startswith('/'):
      new_url = '/' + new_url
    return '![](' + str(new_url) + ')'
  return url_replace_helper

if __name__ == '__main__':
  main()