# CLAUDE.md

## Context
This is a clean project folder, but this is NOT a greenfield ideation project.

A lot of prior reasoning, prototypes, partial code, and business logic already exist from previous work by Carlo with ChatGPT and Claude.
Your task is NOT to reinvent the project from scratch.
Your task is to rebuild the MVP cleanly, using prior knowledge selectively and intelligently.

## Main objective
Build a local usable MVP program for industrial BESS simulation and business case generation.

## Core business question
The tool must answer:

**If I install a BESS here, how can I use it, what value does it generate, and what size makes sense?**

## Target users
- Carlo
- HCE commercial team
- installers
- industrial end customers
- later, possibly financiers / leasing counterparties

## MVP mission
The program must:
1. read a standardized case file
2. simulate BESS scenarios
3. show technical seasonal behavior
4. calculate technical KPIs
5. calculate economic KPIs
6. produce a business plan with payback, NPV, IRR

## Scenarios
- S1 baseline without PV
- S2 current state with PV
- S3 BESS for self-consumption
- S4 BESS multilayer: self-consumption + peak shaving + shifting

## Mandatory output areas
### Technical
- seasonal weekly charts
- load / PV / BESS charge / BESS discharge / grid draw / SOC
- annual battery throughput
- equivalent cycles
- peak shaving effect
- self-consumption effect

### Economic
Keep these always separate:
- saving FV via BESS
- saving reduction of power charge / quota potenza
- margin from energy shifting

Also provide:
- annual net saving
- payback
- NPV
- IRR

## Critical modelling principles
- BESS is a real technical asset, not an abstract energy box
- always account for:
  - nominal capacity
  - usable capacity
  - charge/discharge power
  - SOC min/max
  - round-trip efficiency
  - equivalent annual cycles
  - operational constraints
- distinguish theoretical optimal size vs commercial market size
- weekly seasonal views are for visualization
- annual slot-by-slot simulation is the basis for KPI and economics

## Architecture direction
For now, use this structure:

input case file standardizzato -> simulation engine -> results.json -> dashboard

## Current priority
Do NOT expand the product endlessly.
Do NOT reopen strategy loops unless there is a blocking issue.

Priority is:
**ship a working MVP program**

## Out of scope for now
- generic import from arbitrary customer files
- full product catalog
- full incentives engine
- SaaS architecture
- complete banking-grade workflow
- full advanced automatic sizing optimizer

## Use of historical memory / Obsidian
There is useful prior project memory in Obsidian.
Use it selectively.
Do NOT import all old reasoning blindly.
Extract only what supports the current MVP.

## First working rule
Before major coding:
1. understand the project foundation
2. summarize the MVP scope
3. define the file structure
4. identify useful prior materials
5. only then implement

## Working style
- be surgical
- be honest about open issues
- avoid overengineering
- prefer a shipped MVP over a perfect framework
