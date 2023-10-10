local codeBlock = require('mintlify_utils').codeBlock

local reactPreamble = pandoc.List()

function capitalizeFirstLetter(str)
  return (str:gsub("^%l", string.upper))
end

function castToMintlifyCallout(str)
  if str == "caution" or str == "danger" then
    return "Warning"
  else
    return capitalizeFirstLetter(str)
  end
end

local function addPreamble(preamble)
  if not reactPreamble:includes(preamble) then
    reactPreamble:insert(preamble)
  end
end

local function jsx(content)
  return pandoc.RawBlock("markdown", content)
end

local function tabset(node, filter)
  -- note groupId
  local groupId = ""
  local group = node.attr.attributes["group"]
  if group then
    groupId = ([[ groupId="%s"]]):format(group)
  end

  -- create tabs
  local tabs = pandoc.Div({})
  tabs.content:insert(jsx("<Tabs" .. groupId .. ">"))

  -- iterate through content
  for i = 1, #node.tabs do
    local content = node.tabs[i].content
    local title = node.tabs[i].title

    tabs.content:insert(jsx(([[<TabItem value="%s">]]):format(pandoc.utils.stringify(title))))
    local result = quarto._quarto.ast.walk(content, filter)
    if type(result) == "table" then
      tabs.content:extend(result)
    else
      tabs.content:insert(result)
    end
    tabs.content:insert(jsx("</TabItem>"))
  end

  -- end tab and tabset
  tabs.content:insert(jsx("</Tabs>"))

  -- ensure we have required deps
  addPreamble("import Tabs from '@theme/Tabs';")
  addPreamble("import TabItem from '@theme/TabItem';")

  return tabs
end

function Writer(doc, opts)
  local filter
  filter = {
    CodeBlock = codeBlock,

    DecoratedCodeBlock = function(node)
      local el = node.code_block
      return codeBlock(el, node.filename)
    end,

    Tabset = function(node)
      return tabset(node, filter)
    end,

    RawBlock = function (rawBlock)
      -- We just "pass-through" raw blocks of type "confluence"
      if(rawBlock.format == 'plotly') then
        quarto.utils.dump("Plotly in filter")
        return pandoc.RawBlock('html', rawBlock.text)
      end

      -- Raw blocks inclding arbirtary HTML like JavaScript are not supported in CSF
      return ""
    end,

    Callout = function(node)
      local admonition = pandoc.Div({})
      local mintlifyCallout = castToMintlifyCallout(node.type)
      admonition.content:insert(jsx("<" .. mintlifyCallout .. ">"))
      if node.title then
        admonition.content:insert(pandoc.Header(3, node.title))                
      end
      local content = node.content
      if type(content) == "table" then
        admonition.content:extend(content)
      else
        admonition.content:insert(content)
      end
      admonition.content:insert(jsx("</" .. mintlifyCallout .. ">"))
      return admonition
    end
  }

  doc = quarto._quarto.ast.walk(doc, filter)

  -- insert react preamble if we have it
  if #reactPreamble > 0 then
    local preamble = table.concat(reactPreamble, "\n")
    doc.blocks:insert(1, pandoc.RawBlock("markdown", preamble .. "\n"))
  end

  local extensions = {
    yaml_metadata_block = true,
    pipe_tables = true,
    footnotes = true,
    tex_math_dollars = true,
    raw_html = true,
    all_symbols_escapable = true,
    backtick_code_blocks = true,
    space_in_atx_header = true,
    intraword_underscores = true,
    lists_without_preceding_blankline = true,
    shortcut_reference_links = true,
  }

  return pandoc.write(doc, {
    format = 'markdown_strict',
    extensions = extensions
  }, opts)
end
