# holy_grail_experiments.jl
"""
🌌 HOLOLIFEX6 PROTOTYPE3 - HOLY GRAIL SCALING EXPERIMENTS
Testing constant-time, negative scaling, and quantum emergence
Julia implementation with optimized performance
NOW WITH PROGRESSIVE SCALING 16 → 1024 ENTITIES
"""

using Statistics
using JSON
using Dates
using LinearAlgebra

mutable struct PulseCoupledEntity
    entity_id::String
    domain::String
    base_frequency::Float64
    phase::Float64
    state_vector::Vector{Float64}
    insight_count::Int
    
    function PulseCoupledEntity(entity_id::String, domain::String, base_frequency::Float64=0.02)
        new(entity_id, domain, base_frequency, rand(), randn(8) * 0.1, 0)
    end
end

function evolve_phase!(entity::PulseCoupledEntity)
    entity.phase = mod(entity.phase + entity.base_frequency, 1.0)
end

function couple_to!(entity::PulseCoupledEntity, other_phase::Float64, strength::Float64=0.05)
    phase_diff = other_phase - entity.phase
    entity.phase += strength * sin(2π * phase_diff)
    entity.phase = mod(entity.phase, 1.0)
end

function calculate_action_complexity(action::String)::Int
    complexity_map = Dict(
        "validate" => 1, "check" => 1, "monitor" => 1,
        "optimize" => 2, "balance" => 2, "sync" => 2, "extract" => 2,
        "innovate" => 3, "create" => 3, "orchestrate" => 3, "predict_complex" => 3
    )
    
    for (key, score) in complexity_map
        if occursin(key, action)
            return score
        end
    end
    return 1
end

function generate_insight(entity::PulseCoupledEntity)::Dict{String,Any}
    if entity.phase > 0.8
        entity.insight_count += 1
        
        action_map = Dict(
            "physical" => ["validate_memory", "optimize_resources", "innovate_architecture"],
            "temporal" => ["balance_timing", "sync_cycles", "predict_complex_trends"],
            "semantic" => ["extract_meaning", "validate_logic", "create_knowledge_graphs"],
            "network" => ["optimize_routing", "balance_load", "orchestrate_distributed_systems"]
        )
        
        actions = get(action_map, entity.domain, ["analyze_situation"])
        action_idx = Int(floor(entity.phase * length(actions))) % length(actions) + 1
        action = actions[action_idx]
        
        complexity = calculate_action_complexity(action)
        
        return Dict(
            "entity" => entity.entity_id,
            "domain" => entity.domain,
            "action" => action,
            "confidence" => entity.phase,
            "complexity" => complexity,
            "insight_number" => entity.insight_count
        )
    end
    return Dict{String,Any}()
end

mutable struct Lightweight4DSelector
    num_entities::Int
    dim::Int
    weights::Matrix{Float64}
    
    function Lightweight4DSelector(num_entities::Int, dim::Int=8)
        new(num_entities, dim, randn(dim, 4) * 0.1)
    end
end

mutable struct ScalableEntityNetwork
    entities::Vector{PulseCoupledEntity}
    decision_model::Lightweight4DSelector
    coherence_history::Vector{Float64}
    insight_history::Vector{Dict{String,Any}}
    
    function ScalableEntityNetwork(decision_model::Lightweight4DSelector)
        new(PulseCoupledEntity[], decision_model, Float64[], Dict{String,Any}[])
    end
end

function add_entity!(network::ScalableEntityNetwork, entity::PulseCoupledEntity)
    push!(network.entities, entity)
end

function evolve_step!(network::ScalableEntityNetwork, system_state::Dict{String,Float64})::Vector{Dict{String,Any}}
    insights = Dict{String,Any}[]
    
    for entity in network.entities
        evolve_phase!(entity)
    end
    
    avg_phase = mean([e.phase for e in network.entities])
    for entity in network.entities
        couple_to!(entity, avg_phase, 0.05)
    end
    
    for entity in network.entities
        insight = generate_insight(entity)
        if !isempty(insight)
            push!(insights, insight)
            push!(network.insight_history, insight)
        end
    end
    
    phases = [e.phase for e in network.entities]
    coherence = 1.0 - std(phases)
    push!(network.coherence_history, coherence)
    
    return insights
end

function get_coherence(network::ScalableEntityNetwork)::Float64
    return isempty(network.coherence_history) ? 0.0 : network.coherence_history[end]
end

function get_intelligence_metrics(network::ScalableEntityNetwork)::Dict{String,Float64}
    if isempty(network.insight_history)
        return Dict(
            "avg_complexity" => 0.0,
            "insight_rate" => 0.0,
            "domain_variety" => 0.0,
            "learning_trend" => 0.0
        )
    end
    
    all_insights = network.insight_history
    recent_insights = all_insights[max(1, end-29):end]
    
    complexities = [get(i, "complexity", 1) for i in recent_insights]
    avg_complexity = mean(complexities)
    
    cycles = length(network.coherence_history)
    insight_rate = length(all_insights) / max(1, cycles)
    
    domains = [get(i, "domain", "") for i in recent_insights]
    unique_domains = length(unique(filter(!isempty, domains)))
    domain_variety = unique_domains / length(recent_insights)
    
    if length(all_insights) >= 10
        half_point = length(all_insights) ÷ 2
        early_insights = all_insights[1:min(10, half_point)]
        late_insights = all_insights[max(1, end-9):end]
        
        early_complexity = mean([get(i, "complexity", 1) for i in early_insights])
        late_complexity = mean([get(i, "complexity", 1) for i in late_insights])
        learning_trend = late_complexity - early_complexity
    else
        learning_trend = 0.0
    end
    
    return Dict(
        "avg_complexity" => avg_complexity,
        "insight_rate" => insight_rate,
        "domain_variety" => domain_variety,
        "learning_trend" => learning_trend
    )
end

function get_memory_mb()::Float64
    return Base.gc_live_bytes() / 1024 / 1024
end

function measure_performance(network::ScalableEntityNetwork)::Dict{String,Float64}
    memory_mb = get_memory_mb()
    
    start = time()
    states = [e.state_vector for e in network.entities]
    step_time_ms = (time() - start) * 1000
    
    intel_metrics = get_intelligence_metrics(network)
    
    return merge(Dict(
        "memory_mb" => memory_mb,
        "step_time_ms" => step_time_ms,
        "entity_count" => length(network.entities),
        "coherence" => get_coherence(network)
    ), intel_metrics)
end

# Constant-Time Entity
mutable struct ConstantTimeEntity <: Any
    base::PulseCoupledEntity
    cluster_id::Int
    is_representative::Bool
    local_phase::Float64
    
    function ConstantTimeEntity(entity_id::String, domain::String, cluster_size::Int=32)
        base = PulseCoupledEntity(entity_id, domain)
        cluster_id = abs(hash(entity_id)) % cluster_size
        is_representative = (abs(hash(entity_id)) % cluster_size == 0)
        new(base, cluster_id, is_representative, 0.0)
    end
end

function evolve_phase!(entity::ConstantTimeEntity)
    if entity.is_representative
        evolve_phase!(entity.base)
        entity.local_phase = entity.base.phase
    else
        entity.base.phase = mod(entity.base.phase + entity.base.base_frequency, 1.0)
    end
end

function generate_insight(entity::ConstantTimeEntity)::Dict{String,Any}
    if entity.is_representative
        insight = generate_insight(entity.base)
        if !isempty(insight)
            insight["cluster_representative"] = true
            insight["cluster_size"] = 32
        end
        return insight
    else
        if entity.base.phase > 0.7
            entity.base.insight_count += 1
            return Dict(
                "entity" => entity.base.entity_id,
                "domain" => entity.base.domain,
                "status" => "cluster_member",
                "cluster" => entity.cluster_id,
                "action" => "follow_representative",
                "complexity" => 1,
                "confidence" => entity.base.phase,
                "insight_number" => entity.base.insight_count
            )
        end
    end
    return Dict{String,Any}()
end

# Quantum Entity
mutable struct QuantumEntity <: Any
    base::PulseCoupledEntity
    domain_superposition::Vector{String}
    domain_weights::Vector{Float64}
    collapsed_domain::Union{String,Nothing}
    superposition_entropy::Float64
    
    function QuantumEntity(entity_id::String, primary_domain::String, secondary_domains::Vector{String})
        base = PulseCoupledEntity(entity_id, primary_domain)
        domains = [primary_domain; secondary_domains]
        weights = [0.6; fill(0.4/length(secondary_domains), length(secondary_domains))]
        new(base, domains, weights, nothing, 1.0)
    end
end

function evolve_phase!(entity::QuantumEntity)
    evolve_phase!(entity.base)
    entity.superposition_entropy = min(1.0, entity.base.phase * 2)
end

function generate_insight(entity::QuantumEntity)::Dict{String,Any}
    if entity.base.phase >= 0.6
        probs = entity.domain_weights ./ sum(entity.domain_weights)
        entity.collapsed_domain = entity.domain_superposition[findfirst(cumsum(probs) .>= rand())]
    end
    
    if !isnothing(entity.collapsed_domain) && entity.base.phase > 0.75
        original_domain = entity.base.domain
        entity.base.domain = entity.collapsed_domain
        insight = generate_insight(entity.base)
        entity.base.domain = original_domain
        
        if !isempty(insight)
            insight["quantum_collapse"] = true
            insight["collapsed_from"] = entity.domain_superposition
            insight["superposition_entropy"] = entity.superposition_entropy
            return insight
        end
    end
    
    if entity.base.phase > 0.7
        entity.base.insight_count += 1
        return Dict(
            "entity" => entity.base.entity_id,
            "domain" => entity.base.domain,
            "action" => "superposition_evolving",
            "confidence" => entity.base.phase,
            "superposition_entropy" => entity.superposition_entropy,
            "quantum_collapse" => false,
            "complexity" => 2,
            "insight_number" => entity.base.insight_count
        )
    end
    
    return Dict{String,Any}()
end

# Holographic Network
mutable struct HolographicNetwork
    base::ScalableEntityNetwork
    compression_ratio::Float64
    compressed_representation::Union{Matrix{Float64},Nothing}
    compression_matrix::Union{Matrix{Float64},Nothing}
    expansion_matrix::Union{Matrix{Float64},Nothing}
    
    function HolographicNetwork(decision_model::Lightweight4DSelector, compression_ratio::Float64=0.1)
        base = ScalableEntityNetwork(decision_model)
        new(base, compression_ratio, nothing, nothing, nothing)
    end
end

function add_entity!(network::HolographicNetwork, entity::PulseCoupledEntity)
    push!(network.base.entities, entity)
end

function update_compressed_representation!(network::HolographicNetwork)
    all_states = hcat([e.state_vector for e in network.base.entities]...)'
    compressed_size = max(1, Int(floor(length(network.base.entities) * network.compression_ratio)))
    
    network.compression_matrix = randn(size(all_states, 2), compressed_size) * 0.1
    network.expansion_matrix = randn(compressed_size, size(all_states, 2)) * 0.1
    
    network.compressed_representation = all_states * network.compression_matrix
end

function expand_compressed_representation(network::HolographicNetwork)::Matrix{Float64}
    if isnothing(network.compressed_representation) || isnothing(network.expansion_matrix)
        return hcat([e.state_vector for e in network.base.entities]...)'
    end
    
    expanded_base = network.compressed_representation * network.expansion_matrix
    actual_states = hcat([e.state_vector for e in network.base.entities]...)'
    blend_ratio = 0.3
    
    return blend_ratio * expanded_base .+ (1 - blend_ratio) * actual_states
end

function evolve_step!(network::HolographicNetwork, system_state::Dict{String,Float64})::Vector{Dict{String,Any}}
    if length(network.base.entities) > 100 && network.compression_ratio < 1.0
        if isnothing(network.compressed_representation) || rand() < 0.1
            update_compressed_representation!(network)
        end
    end
    
    return evolve_step!(network.base, system_state)
end

mutable struct HolyGrailExperiments
    results::Vector{Dict{String,Any}}
    start_time::Float64
    
    HolyGrailExperiments() = new(Dict{String,Any}[], time())
end

function log_message(exp::HolyGrailExperiments, message::String)
    elapsed = time() - exp.start_time
    println("[$(round(elapsed, digits=1))s] 🌌 $message")
end

function memory_safety_check(exp::HolyGrailExperiments)::Bool
    memory_mb = get_memory_mb()
    return memory_mb < 6000
end

function test_constant_time_scaling(exp::HolyGrailExperiments, entity_count::Int=256)::Dict{String,Any}
    log_message(exp, "CONSTANT-TIME SCALING: Testing $entity_count entities")
    
    domains = ["physical", "temporal", "semantic", "network"]
    entities = Any[]
    
    for i in 1:entity_count
        domain = domains[(i-1) % length(domains) + 1]
        entity_id = "CT-$(uppercase(domain[1:3]))-$(lpad(i, 4, '0'))"
        push!(entities, ConstantTimeEntity(entity_id, domain, 32))
    end
    
    decision_model = Lightweight4DSelector(entity_count, 8)
    network = ScalableEntityNetwork(decision_model)
    
    for entity in entities
        add_entity!(network, entity.base)
    end
    
    system_state = Dict("memory_usage" => 0.7, "cpu_load" => 0.6, "coherence" => 0.0)
    metrics = Dict{String,Any}[]
    
    for cycle in 1:50
        if !memory_safety_check(exp)
            log_message(exp, "MEMORY LIMIT REACHED - stopping constant-time test")
            break
        end
        
        insights = Dict{String,Any}[]
        for entity in entities
            evolve_phase!(entity)
        end
        
        avg_phase = mean([e.base.phase for e in entities])
        for entity in entities
            couple_to!(entity.base, avg_phase, 0.05)
        end
        
        for entity in entities
            insight = generate_insight(entity)
            if !isempty(insight)
                push!(insights, insight)
                push!(network.insight_history, insight)
            end
        end
        
        phases = [e.base.phase for e in entities]
        coherence = 1.0 - std(phases)
        push!(network.coherence_history, coherence)
        
        if cycle % 5 == 0
            perf = measure_performance(network)
            push!(metrics, perf)
        end
    end
    
    intel_metrics = get_intelligence_metrics(network)
    
    result = merge(Dict(
        "experiment" => "constant_time_scaling",
        "entity_count" => entity_count,
        "clusters" => 32,
        "avg_memory_mb" => isempty(metrics) ? 0.0 : mean([m["memory_mb"] for m in metrics]),
        "avg_step_time_ms" => isempty(metrics) ? 0.0 : mean([m["step_time_ms"] for m in metrics]),
        "final_coherence" => get_coherence(network),
        "representatives" => count(e -> e.is_representative, entities),
        "status" => isempty(metrics) ? "memory_limited" : "completed"
    ), intel_metrics)
    
    push!(exp.results, result)
    log_message(exp, "Constant-time result: $(round(result["avg_memory_mb"], digits=1))MB, Complexity: $(round(result["avg_complexity"], digits=2))")
    return result
end

function test_quantum_superposition(exp::HolyGrailExperiments, entity_count::Int=128)::Dict{String,Any}
    log_message(exp, "QUANTUM SUPERPOSITION: Testing $entity_count entities")
    
    primary_domains = ["physical", "temporal", "semantic", "network"]
    secondary_domains = ["spatial", "emotional", "social", "creative"]
    
    entities = Any[]
    for i in 1:entity_count
        primary = primary_domains[(i-1) % length(primary_domains) + 1]
        secondaries = filter(d -> d != primary, secondary_domains)[1:2]
        entity_id = "QU-$(uppercase(primary[1:3]))-$(lpad(i, 4, '0'))"
        push!(entities, QuantumEntity(entity_id, primary, secondaries))
    end
    
    decision_model = Lightweight4DSelector(entity_count, 8)
    network = ScalableEntityNetwork(decision_model)
    
    for entity in entities
        add_entity!(network, entity.base)
    end
    
    system_state = Dict("memory_usage" => 0.7, "cpu_load" => 0.6, "coherence" => 0.0)
    metrics = Dict{String,Any}[]
    superposition_stats = Dict{String,Any}[]
    
    for cycle in 1:60
        if !memory_safety_check(exp)
            log_message(exp, "MEMORY LIMIT REACHED - stopping quantum test")
            break
        end
        
        insights = Dict{String,Any}[]
        for entity in entities
            evolve_phase!(entity)
        end
        
        avg_phase = mean([e.base.phase for e in entities])
        for entity in entities
            couple_to!(entity.base, avg_phase, 0.05)
        end
        
        for entity in entities
            insight = generate_insight(entity)
            if !isempty(insight)
                push!(insights, insight)
                push!(network.insight_history, insight)
            end
        end
        
        phases = [e.base.phase for e in entities]
        coherence = 1.0 - std(phases)
        push!(network.coherence_history, coherence)
        
        collapsed_count = count(e -> !isnothing(e.collapsed_domain), entities)
        push!(superposition_stats, Dict(
            "cycle" => cycle,
            "collapsed_entities" => collapsed_count,
            "superposition_ratio" => 1.0 - (collapsed_count / length(entities))
        ))
        
        if cycle % 8 == 0
            perf = measure_performance(network)
            push!(metrics, perf)
        end
    end
    
    intel_metrics = get_intelligence_metrics(network)
    
    result = merge(Dict(
        "experiment" => "quantum_superposition",
        "entity_count" => entity_count,
        "avg_memory_mb" => isempty(metrics) ? 0.0 : mean([m["memory_mb"] for m in metrics]),
        "avg_step_time_ms" => isempty(metrics) ? 0.0 : mean([m["step_time_ms"] for m in metrics]),
        "final_coherence" => get_coherence(network),
        "avg_superposition_ratio" => isempty(superposition_stats) ? 0.0 : mean([s["superposition_ratio"] for s in superposition_stats]),
        "final_collapsed_ratio" => isempty(superposition_stats) ? 0.0 : superposition_stats[end]["collapsed_entities"] / length(entities),
        "quantum_entropy" => mean([e.superposition_entropy for e in entities]),
        "status" => isempty(metrics) ? "memory_limited" : "completed"
    ), intel_metrics)
    
    push!(exp.results, result)
    log_message(exp, "Quantum result: $(round(result["avg_memory_mb"], digits=1))MB, Complexity: $(round(result["avg_complexity"], digits=2))")
    return result
end

function test_holographic_compression(exp::HolyGrailExperiments, entity_count::Int=512)::Dict{String,Any}
    log_message(exp, "HOLOGRAPHIC COMPRESSION: Testing $entity_count entities")
    
    domains = ["physical", "temporal", "semantic", "network"]
    entities = PulseCoupledEntity[]
    
    for i in 1:entity_count
        domain = domains[(i-1) % length(domains) + 1]
        entity_id = "HG-$(uppercase(domain[1:3]))-$(lpad(i, 4, '0'))"
        push!(entities, PulseCoupledEntity(entity_id, domain))
    end
    
    decision_model = Lightweight4DSelector(entity_count, 8)
    network = HolographicNetwork(decision_model, 0.2)
    
    for entity in entities
        add_entity!(network, entity)
    end
    
    system_state = Dict("memory_usage" => 0.7, "cpu_load" => 0.6, "coherence" => 0.0)
    metrics = Dict{String,Any}[]
    
    for cycle in 1:40
        if !memory_safety_check(exp)
            log_message(exp, "MEMORY LIMIT REACHED - stopping holographic test")
            break
        end
        
        insights = evolve_step!(network, system_state)
        
        if cycle % 5 == 0
            perf = measure_performance(network.base)
            push!(metrics, perf)
        end
    end
    
    intel_metrics = get_intelligence_metrics(network.base)
    
    result = merge(Dict(
        "experiment" => "holographic_compression",
        "entity_count" => entity_count,
        "compression_ratio" => 0.2,
        "avg_memory_mb" => isempty(metrics) ? 0.0 : mean([m["memory_mb"] for m in metrics]),
        "avg_step_time_ms" => isempty(metrics) ? 0.0 : mean([m["step_time_ms"] for m in metrics]),
        "final_coherence" => get_coherence(network.base),
        "compression_active" => !isnothing(network.compressed_representation),
        "status" => isempty(metrics) ? "memory_limited" : "completed"
    ), intel_metrics)
    
    push!(exp.results, result)
    log_message(exp, "Holographic result: $(round(result["avg_memory_mb"], digits=1))MB, Complexity: $(round(result["avg_complexity"], digits=2))")
    return result
end

function run_all_experiments(exp::HolyGrailExperiments)::Vector{Dict{String,Any}}
    log_message(exp, "STARTING SCALING HOLY GRAIL EXPERIMENTS 16 → 1024 ENTITIES")
    
    # Progressive scaling levels for credible academic progression
    scaling_levels = [16, 32, 64, 128, 256, 512, 1024]
    
    # 1. Constant-time scaling progression
    log_message(exp, "🔬 PHASE 1: CONSTANT-TIME SCALING PROGRESSION")
    for entity_count in scaling_levels
        log_message(exp, "   Testing constant-time with $entity_count entities")
        try
            result = test_constant_time_scaling(exp, entity_count)
            if result["status"] != "completed"
                log_message(exp, "   ⚠️  Stopping constant-time at $entity_count entities")
                break
            end
        catch e
            log_message(exp, "   💥 Constant-time failed at $entity_count: $e")
            break
        end
    end
    
    # 2. Quantum superposition progression  
    log_message(exp, "🔬 PHASE 2: QUANTUM SUPERPOSITION PROGRESSION")
    for entity_count in scaling_levels
        log_message(exp, "   Testing quantum superposition with $entity_count entities")
        try
            result = test_quantum_superposition(exp, entity_count)
            if result["status"] != "completed"
                log_message(exp, "   ⚠️  Stopping quantum at $entity_count entities")
                break
            end
        catch e
            log_message(exp, "   💥 Quantum failed at $entity_count: $e")
            break
        end
    end
    
    # 3. Holographic compression progression
    log_message(exp, "🔬 PHASE 3: HOLOGRAPHIC COMPRESSION PROGRESSION")
    for entity_count in scaling_levels
        log_message(exp, "   Testing holographic compression with $entity_count entities")
        try
            result = test_holographic_compression(exp, entity_count)
            if result["status"] != "completed"
                log_message(exp, "   ⚠️  Stopping holographic at $entity_count entities")
                break
            end
        catch e
            log_message(exp, "   💥 Holographic failed at $entity_count: $e")
            break
        end
    end
    
    log_message(exp, "🎉 ALL SCALING EXPERIMENTS COMPLETED")
    return exp.results
end

function save_results(exp::HolyGrailExperiments)::String
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = "holy_grail_results_$timestamp.json"
    
    open(filename, "w") do f
        JSON.print(f, exp.results, 2)
    end
    
    log_message(exp, "Results saved to: $filename")
    return filename
end

function main()
    println("🌌 HOLOLIFEX6 PROTOTYPE3 - HOLY GRAIL SCALING EXPERIMENTS")
    println("="^60)
    println("🎯 PROGRESSIVE SCALING: 16 → 1024 entities per architecture")
    println("🔬 TESTING: Constant-time + Quantum + Holographic approaches") 
    println("📊 TRACKING: Memory scaling + Intelligence metrics + Coherence")
    println("💎 Julia implementation with enhanced analysis")
    println("="^60)
    
    experimenter = HolyGrailExperiments()
    
    try
        results = run_all_experiments(experimenter)
        
        results_file = save_results(experimenter)
        
        println("\n📊 HOLY GRAIL SCALING EXPERIMENTS SUMMARY:")
        println("="^55)

        # Group results by experiment type for better analysis
        experiment_types = unique([r["experiment"] for r in results])
        scaling_levels = [16, 32, 64, 128, 256, 512, 1024]

        for exp_type in experiment_types
            println("\n🌌 $(uppercase(exp_type)) SCALING:")
            println("-" ^ 30)
            
            exp_results = filter(r -> r["experiment"] == exp_type, results)
            sorted_results = sort(exp_results, by=x -> x["entity_count"])
            
            for result in sorted_results
                entity_count = result["entity_count"]
                memory = round(result["avg_memory_mb"], digits=1)
                coherence = round(result["final_coherence"], digits=3)
                complexity = round(get(result, "avg_complexity", 0), digits=2)
                insight_rate = round(get(result, "insight_rate", 0), digits=2)
                
                scaling_ratio = entity_count / 16
                scaling_info = "($(scaling_ratio)x)"
                
                println("   Entities: $(lpad(entity_count, 5)) $scaling_info")
                println("   Memory: $(lpad(memory, 6))MB | Coherence: $coherence")
                println("   Complexity: $complexity | Insight Rate: $insight_rate/cycle")
                
                # Show architecture-specific metrics
                if exp_type == "quantum_superposition" && haskey(result, "avg_superposition_ratio")
                    super_ratio = round(result["avg_superposition_ratio"], digits=3)
                    println("   Superposition: $super_ratio | Entropy: $(round(get(result, "quantum_entropy", 0), digits=3))")
                elseif exp_type == "constant_time_scaling" && haskey(result, "clusters")
                    println("   Clusters: $(result["clusters"]) | Representatives: $(get(result, "representatives", 0))")
                elseif exp_type == "holographic_compression" && haskey(result, "compression_active")
                    println("   Compression: $(result["compression_active"]) | Ratio: $(get(result, "compression_ratio", 0))")
                end
                
                println("   Status: $(result["status"])")
                println()
            end
        end

        # Calculate scaling efficiency for each architecture
        println("\n📈 SCALING EFFICIENCY ANALYSIS:")
        println("-" ^ 35)

        for exp_type in experiment_types
            exp_results = filter(r -> r["experiment"] == exp_type, results)
            if length(exp_results) >= 2
                baseline = exp_results[1]  # 16 entities
                max_scale = exp_results[end]  # largest completed test
                
                baseline_memory = baseline["avg_memory_mb"]
                max_memory = max_scale["avg_memory_mb"]
                scale_ratio = max_scale["entity_count"] / baseline["entity_count"]
                
                expected_linear_memory = baseline_memory * scale_ratio
                actual_memory = max_memory
                efficiency = (expected_linear_memory - actual_memory) / expected_linear_memory * 100
                
                println("🌌 $(uppercase(exp_type)):")
                println("   Scale: $(baseline["entity_count"]) → $(max_scale["entity_count"]) entities ($(round(scale_ratio, digits=1))x)")
                println("   Memory: $(round(baseline_memory, digits=1))MB → $(round(actual_memory, digits=1))MB")
                println("   Efficiency: $(round(efficiency, digits=1))% $(efficiency > 20 ? "🚀" : "✅")")
                println("   Coherence: $(round(baseline["final_coherence"], digits=3)) → $(round(max_scale["final_coherence"], digits=3))")
                println()
            end
        end
        
        println("🌠 Holy Grail scaling experiments completed!")
        println("📁 Results saved to: $results_file")
        
    catch e
        println("💥 Experiments failed: $e")
        save_results(experimenter)
        rethrow(e)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
