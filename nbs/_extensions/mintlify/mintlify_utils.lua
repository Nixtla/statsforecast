-- local scriptCount = 0
-- local printItem = 5

function codeBlock(el, filename)
  local lang = el.attr.classes[1]
  -- scriptCount = scriptCount + 1  
  -- if printItem == scriptCount then    
  --   quarto.utils.dump(el) 
  -- end
  -- quarto.log.output('---Code block---') 
  -- quarto.utils.dump(el.output) 
  -- quarto.utils.dump({}) 
  -- quarto.log.output('------') 
  local title = filename or el.attr.attributes["filename"] or el.attr.attributes["title"]  
  local showLineNumbers = el.attr.classes:includes('number-lines')
  if lang or title or showLineNumbers then
    if not lang then
      lang = 'text'
    end
    local code = "\n```" .. lang
    if showLineNumbers then
      code = code .. " showLineNumbers"
    end
    if title then
      code = code .. " title=\"" .. title .. "\""
    end
    code = code .. "\n" .. el.text .. "\n```\n"

    -- quarto.log.output('------') 
    -- quarto.log.output(code) 
    -- quarto.log.output('------') 
    -- docusaures code block attributes don't conform to any syntax
    -- that pandoc natively understands, so return the CodeBlock as
    -- "raw" markdown (so it bypasses pandoc processing entirely)
    return pandoc.RawBlock("markdown", code)

  elseif #el.attr.classes == 0 then
    el.attr.classes:insert('text')
    return el
  end

  return nil
end

return {
  codeBlock = codeBlock
}