param([Parameter(Mandatory = $true)][ValidatePattern('^[A-Za-z0-9._-]{1,128}$')][string]$RunId)
$ErrorActionPreference = 'Stop'
if ($RunId -match '^\.+$') { throw 'Unsafe run id' }
$root = (git rev-parse --show-toplevel).Trim()
$base = Join-Path $root 'results/remote'
$dest = Join-Path $base $RunId
$partial = Join-Path $base ('.partial-' + $RunId + '-' + $PID)
if (Test-Path $dest -PathType Any) { throw "Destination already exists: $dest" }
if (Test-Path $partial -PathType Any) { throw "Partial destination already exists: $partial" }
New-Item -ItemType Directory -Path $partial -Force | Out-Null
try {
  & scp -P 22003 -r "qingchan@asimov1.cs.uga.edu:/home/qingchan/data/concept-flow/runs/$RunId/." "$partial/"
  if ($LASTEXITCODE -ne 0) { throw 'scp failed' }
  $partialRoot = [IO.Path]::GetFullPath($partial) + [IO.Path]::DirectorySeparatorChar
  foreach ($line in Get-Content -LiteralPath (Join-Path $partial 'SHA256SUMS')) {
    if ($line -notmatch '^([0-9a-f]{64})  \./(.+)$') { throw "Invalid checksum line: $line" }
    $expected = $Matches[1]
    $candidate = [IO.Path]::GetFullPath((Join-Path $partial $Matches[2]))
    if (-not $candidate.StartsWith($partialRoot, [StringComparison]::OrdinalIgnoreCase)) {
      throw "Checksum path escapes partial directory: $line"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw "Missing artifact: $candidate" }
    $actual = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Checksum failed: $candidate" }
  }
  if (-not (Test-Path (Join-Path $partial 'metadata.env'))) { throw 'metadata missing' }
  Move-Item -LiteralPath $partial -Destination $dest
} finally {
  if (Test-Path $partial) { Remove-Item -LiteralPath $partial -Recurse -Force }
}
$dest
