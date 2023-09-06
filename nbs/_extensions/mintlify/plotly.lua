-- local scriptCount = 0
-- local printItem = 4
function script(scriptEl)
    -- scriptCount = scriptCount + 1
    -- quarto.log.output('---Script El---')     
    -- quarto.log.output(scriptCount)     
    -- if printItem == scriptCount then    
    --     quarto.utils.dump(scriptEl) 
    -- end
    quarto.log.output('---Script El---')     
end

return {
  {
    ['application/vnd.plotly.v1+json'] = script
  }
}