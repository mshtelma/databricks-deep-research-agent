import { Settings, RefreshCw, Info } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { useSettingsStore } from '@/stores/settingsStore'

export function AgentSettings() {
  const {
    agentConfig,
    isSettingsOpen,
    updateAgentConfig,
    resetToDefaults,
    setSettingsOpen
  } = useSettingsStore()

  const reportStyleOptions = [
    { value: 'professional', label: 'Professional', description: 'Formal, structured reports for business use' },
    { value: 'casual', label: 'Casual', description: 'Conversational style for everyday questions' },
    { value: 'academic', label: 'Academic', description: 'Scholarly format with detailed citations' },
    { value: 'technical', label: 'Technical', description: 'Technical documentation style with specs' }
  ]

  const verificationLevelOptions = [
    { value: 'basic', label: 'Basic', description: 'Quick fact-checking for speed' },
    { value: 'moderate', label: 'Moderate', description: 'Balanced accuracy and performance' },
    { value: 'thorough', label: 'Thorough', description: 'Comprehensive fact verification' }
  ]

  return (
    <Popover open={isSettingsOpen} onOpenChange={setSettingsOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
        >
          <Settings className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-96 p-6" align="end">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">Agent Settings</h3>
              <p className="text-sm text-gray-500">Configure multi-agent research behavior</p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={resetToDefaults}
              className="h-8 w-8 p-0"
              title="Reset to defaults"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>

          {/* Report Style */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Report Style</label>
            <Select
              value={agentConfig.reportStyle}
              onValueChange={(value) => 
                updateAgentConfig({ reportStyle: value as any })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {reportStyleOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    <div className="flex flex-col">
                      <span className="font-medium">{option.label}</span>
                      <span className="text-xs text-gray-500">{option.description}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Verification Level */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Fact Verification Level</label>
            <Select
              value={agentConfig.verificationLevel}
              onValueChange={(value) => 
                updateAgentConfig({ verificationLevel: value as any })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {verificationLevelOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    <div className="flex flex-col">
                      <span className="font-medium">{option.label}</span>
                      <span className="text-xs text-gray-500">{option.description}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Multi-Agent Workflow Settings */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-gray-800 dark:text-gray-200">
              Multi-Agent Workflow
            </h4>
            
            {/* Background Investigation */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <label className="text-sm">Background Investigation</label>
                <div className="group relative">
                  <Info className="h-3 w-3 text-gray-400 cursor-help" />
                  <div className="invisible group-hover:visible absolute bottom-5 left-0 z-50 w-48 p-2 bg-black text-white text-xs rounded shadow-lg">
                    Gather initial context before planning research
                  </div>
                </div>
              </div>
              <Switch
                checked={agentConfig.enableBackgroundInvestigation}
                onCheckedChange={(checked) => 
                  updateAgentConfig({ enableBackgroundInvestigation: checked })
                }
              />
            </div>

            {/* Iterative Planning */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <label className="text-sm">Iterative Planning</label>
                <div className="group relative">
                  <Info className="h-3 w-3 text-gray-400 cursor-help" />
                  <div className="invisible group-hover:visible absolute bottom-5 left-0 z-50 w-48 p-2 bg-black text-white text-xs rounded shadow-lg">
                    Allow planner to revise and improve research plans
                  </div>
                </div>
              </div>
              <Switch
                checked={agentConfig.enableIterativePlanning}
                onCheckedChange={(checked) => 
                  updateAgentConfig({ enableIterativePlanning: checked })
                }
              />
            </div>

            {/* Auto-Accept Plans */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <label className="text-sm">Auto-Accept Plans</label>
                <div className="group relative">
                  <Info className="h-3 w-3 text-gray-400 cursor-help" />
                  <div className="invisible group-hover:visible absolute bottom-5 left-0 z-50 w-48 p-2 bg-black text-white text-xs rounded shadow-lg">
                    Automatically proceed with good quality plans
                  </div>
                </div>
              </div>
              <Switch
                checked={agentConfig.autoAcceptPlan}
                onCheckedChange={(checked) => 
                  updateAgentConfig({ autoAcceptPlan: checked })
                }
              />
            </div>

            {/* Max Plan Iterations */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <label className="text-sm">Max Plan Iterations</label>
                  <Badge variant="secondary" className="text-xs">
                    {agentConfig.maxPlanIterations}
                  </Badge>
                </div>
              </div>
              <Slider
                value={[agentConfig.maxPlanIterations]}
                onValueChange={([value]) => 
                  updateAgentConfig({ maxPlanIterations: value })
                }
                min={1}
                max={5}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>1 (faster)</span>
                <span>5 (thorough)</span>
              </div>
            </div>
          </div>

          {/* Agent Status Indicators */}
          <div className="pt-2 border-t">
            <h4 className="text-sm font-medium text-gray-800 dark:text-gray-200 mb-3">
              Active Agents
            </h4>
            <div className="flex flex-wrap gap-1">
              {[
                { name: 'Coordinator', icon: '🎯', color: 'bg-blue-100 text-blue-800' },
                { name: 'Planner', icon: '📋', color: 'bg-purple-100 text-purple-800' },
                { name: 'Researcher', icon: '🔬', color: 'bg-orange-100 text-orange-800' },
                { name: 'Fact Checker', icon: '🔎', color: 'bg-red-100 text-red-800' },
                { name: 'Reporter', icon: '📄', color: 'bg-green-100 text-green-800' }
              ].map((agent) => (
                <div 
                  key={agent.name}
                  className={`px-2 py-1 rounded-full text-xs font-medium ${agent.color}`}
                >
                  <span className="mr-1">{agent.icon}</span>
                  {agent.name}
                </div>
              ))}
            </div>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}