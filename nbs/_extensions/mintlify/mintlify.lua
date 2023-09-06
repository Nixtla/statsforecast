-- mintlify.lua

local kQuartoRawHtml = "quartoRawHtml"
local rawHtmlVars = pandoc.List()
function Pandoc(doc)
  -- quarto.utils.dump(doc)
  -- quarto.log.output('---E---') 
  -- insert exports at the top if we have them
  if #rawHtmlVars > 0 then
    local exports = ("export const %s =\n[%s];"):format(kQuartoRawHtml, 
      table.concat(
        rawHtmlVars:map(function(var) return '`'.. var .. '`' end), 
        ","
      )
    )
    doc.blocks:insert(1, pandoc.RawBlock("markdown", exports .. "\n"))
  end

  return doc
end


-- strip image attributes (which may result from
-- fig-format: retina) as they will result in an
-- img tag which won't hit the asset pipeline
function Image(el)
  -- quarto.log.output('---D---') 
  el.attr = pandoc.Attr()
  return el
end

-- header attributes only support id
function Header(el)
  -- quarto.log.output('---C---') 
  el.attr = pandoc.Attr(el.identifier)
  return el
end

Block = function(node) 
  if node.text ~= nil and string.find(node.text, "<CustomCode") then      
    quarto.utils.dump("Plotly in global")
    return pandoc.RawBlock("plotly", node.text)      
  end
  return node    
end

-- local scriptCount = 0
-- local printItem = 1

-- transform 'mdx' into passthrough content, transform 'html'
-- into raw commamark to pass through via dangerouslySetInnerHTML
function RawBlock(el)  
  -- quarto.log.output(el.format)
  -- scriptCount = scriptCount + 1  
  if el.format == 'mdx' then
    -- quarto.log.output('---A---') 
    return pandoc.CodeBlock(el.text, pandoc.Attr("", {"mdx-code-block"}))
  elseif el.format == 'html' then
    -- quarto.utils.dump(el.text)
    -- quarto.log.output('---B---') 
    -- if printItem == scriptCount then    
    --   quarto.utils.dump(el) 
    -- end
    -- track the raw html vars (we'll insert them at the top later on as
    -- mdx requires all exports be declared together)
    local html = string.gsub(el.text, "\n+", "\n")
    rawHtmlVars:insert(html)

    -- generate a div container for the raw html and return it as the block
    local html = ("<div dangerouslySetInnerHTML={{ __html: %s[%d] }} />")
      :format(kQuartoRawHtml, #rawHtmlVars-1) .. "\n"
    return pandoc.RawBlock("html", html)
  end
end

-- -- local i = 0
-- function  CodeBlock(doc)
--   -- i = i + 1
--   -- if i == 5 then
--   --   -- quarto.utils.dump(doc.text)
--   -- end
-- end

-- function  DefinitionList(doc)
--   -- quarto.log.output('DefinitionList')
-- end

-- function Methods(doc)
--   -- quarto.log.output('OrderedList')
--   -- quarto.utils.dump(doc)
-- end