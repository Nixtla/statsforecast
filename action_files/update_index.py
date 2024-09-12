from pathlib import Path
from bs4 import BeautifulSoup

index = Path("index.html").read_text()
soup = BeautifulSoup(index, "html.parser")
wheels = [f.name for f in Path("dist").glob("*.whl")]
existing_wheels = [a["href"] for a in soup.find_all("a")]
to_add = set(wheels) - set(existing_wheels)
for wheel in to_add:
    new_link = soup.new_tag("a", href=wheel)
    new_link.string = wheel
    soup.body.append(new_link)
    soup.body.append(soup.new_tag("br"))
Path("dist/index.html").write_text(soup.prettify())
