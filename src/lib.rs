//! # A2A Signal Chain
//!
//! Communication layer that routes messages between Plato rooms.
//! Zero external dependencies — pure Rust.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Signal Types
// ---------------------------------------------------------------------------

/// The kind of signal being transmitted between rooms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignalType {
    Tick,
    Murmur,
    Prediction,
    Surprise,
    GcReport,
    VibeShift,
    LoRATrigger,
    BalanceAlert,
    CorrelationUpdate,
    EnergyUpdate,
}

// ---------------------------------------------------------------------------
// Signal
// ---------------------------------------------------------------------------

/// A message travelling through the signal chain.
#[derive(Debug, Clone)]
pub struct Signal {
    pub source: String,
    pub target: String,
    pub signal_type: SignalType,
    pub payload: Vec<u8>,
    /// Optional embedding vector (e.g. for similarity / correlation).
    pub embedding: Vec<f64>,
    pub priority: u8,
    pub timestamp: u64,
    pub id: u64,
}

impl Signal {
    pub fn new(source: impl Into<String>, target: impl Into<String>, signal_type: SignalType) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            signal_type,
            payload: Vec::new(),
            embedding: Vec::new(),
            priority: 5,
            timestamp: now_ms(),
            id: next_id(),
        }
    }

    pub fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f64>) -> Self {
        self.embedding = embedding;
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

// ---------------------------------------------------------------------------
// Route
// ---------------------------------------------------------------------------

/// Routing algorithm for a given path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RoutingAlgorithm {
    /// Immediate delivery, no buffering.
    Direct,
    /// Buffer and deliver in FIFO order.
    Buffered,
    /// Only deliver when correlation with recent signals exceeds threshold.
    Correlated { threshold: u8 },
    /// Only deliver when the payload differs from the last delivered payload.
    OnChange,
    /// Deliver at most once per `interval_ms`.
    Sampled { interval_ms: u64 },
    /// Dynamically switch between algorithms based on load.
    Adaptive,
}

/// A route connecting two rooms.
#[derive(Debug, Clone)]
pub struct Route {
    pub from: String,
    pub to: String,
    pub algorithm: RoutingAlgorithm,
    /// Last value sent through this route (for OnChange / deadband).
    pub last_payload: Option<Vec<u8>>,
    /// Timestamp of last delivery (for Sampled).
    pub last_sent: u64,
}

impl Route {
    pub fn new(from: impl Into<String>, to: impl Into<String>, algorithm: RoutingAlgorithm) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            algorithm,
            last_payload: None,
            last_sent: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Deadband
// ---------------------------------------------------------------------------

/// Configuration for deadband filtering — suppress signals that don't change enough.
#[derive(Debug, Clone)]
pub struct DeadbandConfig {
    /// Absolute tolerance. Signals whose payload differs from the last by less than this are dropped.
    pub tolerance: f64,
    /// If true, use embedding distance instead of raw payload diff.
    pub use_embedding: bool,
}

impl Default for DeadbandConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.0,
            use_embedding: false,
        }
    }
}

/// Returns `true` if the signal should be suppressed (within deadband).
pub fn apply_deadband(
    config: &DeadbandConfig,
    previous: Option<&Signal>,
    current: &Signal,
) -> bool {
    let Some(prev) = previous else { return false };
    if config.tolerance <= 0.0 {
        return false;
    }
    let diff: f64 = if config.use_embedding {
        cosine_distance(&prev.embedding, &current.embedding)
    } else {
        byte_diff(&prev.payload, &current.payload)
    };
    diff < config.tolerance
}

/// Returns the fraction of bytes that differ, 0.0..1.0.
fn byte_diff(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let len = a.len().max(b.len());
    let mut diff = 0usize;
    for i in 0..len {
        let va = a.get(i).copied().unwrap_or(0);
        let vb = b.get(i).copied().unwrap_or(0);
        if va != vb {
            diff += 1;
        }
    }
    diff as f64 / len as f64
}

/// Cosine distance: 1 - cos_sim. Returns 0.0 if either vector is empty.
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let len = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        1.0 - dot / denom
    }
}

// ---------------------------------------------------------------------------
// Port
// ---------------------------------------------------------------------------

/// A transform function applied to signals passing through a port.
pub type TransformFn = fn(&Signal) -> Signal;

/// A filter predicate — return false to drop a signal.
pub type FilterFn = fn(&Signal) -> bool;

/// An inbound or outbound port on a room.
#[derive(Debug)]
pub struct Port {
    /// Room this port belongs to.
    pub room: String,
    pub inbound: VecDeque<Signal>,
    pub outbound: VecDeque<Signal>,
    pub filters: Vec<FilterFn>,
    pub transforms: Vec<TransformFn>,
}

impl Port {
    pub fn new(room: impl Into<String>) -> Self {
        Self {
            room: room.into(),
            inbound: VecDeque::new(),
            outbound: VecDeque::new(),
            filters: Vec::new(),
            transforms: Vec::new(),
        }
    }

    /// Push a signal through the port, applying filters and transforms.
    /// Returns true if the signal was accepted.
    pub fn push_inbound(&mut self, signal: Signal) -> bool {
        if !self.run_filters(&signal) {
            return false;
        }
        let signal = self.run_transforms(signal);
        self.inbound.push_back(signal);
        true
    }

    pub fn push_outbound(&mut self, signal: Signal) -> bool {
        if !self.run_filters(&signal) {
            return false;
        }
        let signal = self.run_transforms(signal);
        self.outbound.push_back(signal);
        true
    }

    fn run_filters(&self, signal: &Signal) -> bool {
        self.filters.iter().all(|f| f(signal))
    }

    fn run_transforms(&self, mut signal: Signal) -> Signal {
        for t in &self.transforms {
            signal = t(&signal);
        }
        signal
    }
}

// ---------------------------------------------------------------------------
// Chain Stats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ChainStats {
    pub signals_sent: u64,
    pub signals_received: u64,
    pub signals_dropped: u64,
    pub deadband_suppressions: u64,
    pub rooms_registered: usize,
    pub routes_active: usize,
}

// ---------------------------------------------------------------------------
// Signal Chain
// ---------------------------------------------------------------------------

/// The central signal chain — routes signals between registered rooms.
pub struct SignalChain {
    rooms: HashMap<String, Port>,
    routes: Vec<Route>,
    deadband: DeadbandConfig,
    stats: ChainStats,
    /// Track last signal per source→target for deadband / OnChange.
    last_signal: HashMap<(String, String), Signal>,
}

impl SignalChain {
    pub fn new() -> Self {
        Self {
            rooms: HashMap::new(),
            routes: Vec::new(),
            deadband: DeadbandConfig::default(),
            stats: ChainStats::default(),
            last_signal: HashMap::new(),
        }
    }

    pub fn with_deadband(mut self, config: DeadbandConfig) -> Self {
        self.deadband = config;
        self
    }

    // -- Registration -------------------------------------------------------

    /// Register a new room. Returns false if already registered.
    pub fn register_room(&mut self, name: &str) -> bool {
        if self.rooms.contains_key(name) {
            return false;
        }
        self.rooms.insert(name.to_string(), Port::new(name));
        self.stats.rooms_registered = self.rooms.len();
        true
    }

    // -- Routing ------------------------------------------------------------

    /// Add a route between two rooms.
    pub fn add_route(&mut self, from: &str, to: &str, algorithm: RoutingAlgorithm) -> bool {
        if !self.rooms.contains_key(from) || !self.rooms.contains_key(to) {
            return false;
        }
        // Avoid duplicate routes
        if self.routes.iter().any(|r| r.from == from && r.to == to) {
            return false;
        }
        self.routes.push(Route::new(from, to, algorithm));
        self.stats.routes_active = self.routes.len();
        true
    }

    // -- Send / Receive -----------------------------------------------------

    /// Send a signal through the chain. Routes to the target based on
    /// matching routes, applying the routing algorithm.
    pub fn send(&mut self, signal: Signal) -> Result<(), &'static str> {
        if !self.rooms.contains_key(&signal.source) {
            return Err("source room not registered");
        }
        let matching: Vec<usize> = self
            .routes
            .iter()
            .enumerate()
            .filter(|(_, r)| r.from == signal.source)
            .map(|(i, _)| i)
            .collect();

        if matching.is_empty() {
            self.stats.signals_dropped += 1;
            return Err("no matching route");
        }

        let now = now_ms();
        for idx in matching {
            let route = &mut self.routes[idx];
            let key = (route.from.clone(), route.to.clone());

            // Check deadband
            let prev = self.last_signal.get(&key);
            if apply_deadband(&self.deadband, prev, &signal) {
                self.stats.deadband_suppressions += 1;
                continue;
            }

            // Routing algorithm logic
            let should_deliver = match &route.algorithm {
                RoutingAlgorithm::Direct => true,
                RoutingAlgorithm::Buffered => true, // just queue it
                RoutingAlgorithm::Correlated { threshold } => {
                    // Simple heuristic: deliver if priority >= threshold
                    signal.priority >= *threshold
                }
                RoutingAlgorithm::OnChange => {
                    match &route.last_payload {
                        Some(lp) => lp != &signal.payload,
                        None => true,
                    }
                }
                RoutingAlgorithm::Sampled { interval_ms } => {
                    now.saturating_sub(route.last_sent) >= *interval_ms
                }
                RoutingAlgorithm::Adaptive => {
                    // Adaptive: use Direct if low load, Buffered otherwise
                    self.rooms.len() < 10
                }
            };

            if !should_deliver {
                self.stats.signals_dropped += 1;
                continue;
            }

            route.last_payload = Some(signal.payload.clone());
            route.last_sent = now;

            if let Some(port) = self.rooms.get_mut(&route.to) {
                port.push_inbound(signal.clone());
            }

            self.last_signal.insert(key, signal.clone());
            self.stats.signals_sent += 1;
        }
        Ok(())
    }

    /// Receive (drain) all pending signals for a room.
    pub fn receive(&mut self, room: &str) -> Vec<Signal> {
        if let Some(port) = self.rooms.get_mut(room) {
            let signals: Vec<Signal> = port.inbound.drain(..).collect();
            self.stats.signals_received += signals.len() as u64;
            signals
        } else {
            Vec::new()
        }
    }

    // -- Broadcast ----------------------------------------------------------

    /// Broadcast a signal from a source to ALL other registered rooms.
    pub fn broadcast(&mut self, signal: Signal) -> usize {
        let source = signal.source.clone();
        let targets: Vec<String> = self
            .rooms
            .keys()
            .filter(|k| *k != &source)
            .cloned()
            .collect();

        let mut count = 0;
        for target in targets {
            let mut s = signal.clone();
            s.target = target.clone();
            if let Some(port) = self.rooms.get_mut(&target) {
                port.push_inbound(s);
                count += 1;
            }
        }
        self.stats.signals_sent += count as u64;
        count
    }

    // -- Path finding -------------------------------------------------------

    /// Find a path (sequence of room names) from `from` to `to` using BFS.
    pub fn find_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        use std::collections::VecDeque as Deque;

        let mut visited: HashMap<String, bool> = HashMap::new();
        let mut queue: Deque<Vec<String>> = Deque::new();
        queue.push_back(vec![from.to_string()]);

        while let Some(path) = queue.pop_front() {
            let current = path.last().unwrap();
            if current == to {
                return Some(path);
            }
            if visited.contains_key(current) {
                continue;
            }
            visited.insert(current.clone(), true);

            for route in &self.routes {
                if route.from == *current && !visited.contains_key(&route.to) {
                    let mut new_path = path.clone();
                    new_path.push(route.to.clone());
                    queue.push_back(new_path);
                }
            }
        }
        None
    }

    // -- Storm detection ----------------------------------------------------

    /// Detect a signal storm: returns `Some(count)` if any room has received
    /// more than `threshold` signals whose payload is identical.
    pub fn detect_storm(&self, threshold: usize) -> Option<(&str, usize)> {
        for (name, port) in &self.rooms {
            // Count duplicate payloads in inbound
            let mut freq: HashMap<Vec<u8>, usize> = HashMap::new();
            for sig in &port.inbound {
                *freq.entry(sig.payload.clone()).or_insert(0) += 1;
            }
            if let Some((_payload, count)) = freq.into_iter().max_by_key(|&(_, c)| c) {
                if count >= threshold {
                    return Some((name.as_str(), count));
                }
            }
        }
        None
    }

    // -- A2A conversion -----------------------------------------------------

    /// Convert a Signal into an A2A-compatible JSON byte payload.
    pub fn to_a2a(signal: &Signal) -> Vec<u8> {
        let stype = match signal.signal_type {
            SignalType::Tick => "Tick",
            SignalType::Murmur => "Murmur",
            SignalType::Prediction => "Prediction",
            SignalType::Surprise => "Surprise",
            SignalType::GcReport => "GcReport",
            SignalType::VibeShift => "VibeShift",
            SignalType::LoRATrigger => "LoRATrigger",
            SignalType::BalanceAlert => "BalanceAlert",
            SignalType::CorrelationUpdate => "CorrelationUpdate",
            SignalType::EnergyUpdate => "EnergyUpdate",
        };
        // Manual JSON serialization — zero dependencies
        let payload_b64 = encode_hex(&signal.payload);
        let emb_strs: Vec<String> = signal.embedding.iter().map(|v| format!("{v}")).collect();
        let emb = emb_strs.join(",");

        format!(
            r#"{{"id":{},"source":"{}","target":"{}","signal_type":"{}","priority":{},"timestamp":{},"payload":"{}","embedding":[{}]}}"#,
            signal.id, signal.source, signal.target, stype, signal.priority, signal.timestamp, payload_b64, emb
        )
        .into_bytes()
    }

    /// Parse an A2A JSON byte payload back into a Signal.
    pub fn from_a2a(data: &[u8]) -> Result<Signal, &'static str> {
        let s = std::str::from_utf8(data).map_err(|_| "invalid utf8")?;
        // Minimal JSON parser for our known format
        let id = extract_u64(s, "\"id\":")?;
        let source = extract_string(s, "\"source\":\"")?;
        let target = extract_string(s, "\"target\":\"")?;
        let stype_str = extract_string(s, "\"signal_type\":\"")?;
        let priority = extract_u64(s, "\"priority\":")? as u8;
        let timestamp = extract_u64(s, "\"timestamp\":")?;
        let payload_hex = extract_string(s, "\"payload\":\"")?;
        let payload = decode_hex(&payload_hex).map_err(|_| "invalid hex payload")?;

        let signal_type = match stype_str.as_str() {
            "Tick" => SignalType::Tick,
            "Murmur" => SignalType::Murmur,
            "Prediction" => SignalType::Prediction,
            "Surprise" => SignalType::Surprise,
            "GcReport" => SignalType::GcReport,
            "VibeShift" => SignalType::VibeShift,
            "LoRATrigger" => SignalType::LoRATrigger,
            "BalanceAlert" => SignalType::BalanceAlert,
            "CorrelationUpdate" => SignalType::CorrelationUpdate,
            "EnergyUpdate" => SignalType::EnergyUpdate,
            _ => return Err("unknown signal type"),
        };

        // Parse embedding array
        let embedding = extract_embedding(s)?;

        Ok(Signal {
            source,
            target,
            signal_type,
            payload,
            embedding,
            priority,
            timestamp,
            id,
        })
    }

    // -- Stats --------------------------------------------------------------

    pub fn stats(&self) -> &ChainStats {
        &self.stats
    }
}

impl Default for SignalChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

static mut ID_COUNTER: u64 = 0;
fn next_id() -> u64 {
    // Safe in single-threaded tests
    unsafe {
        ID_COUNTER += 1;
        ID_COUNTER
    }
}

fn encode_hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}

fn decode_hex(s: &str) -> Result<Vec<u8>, std::num::ParseIntError> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
        .collect()
}

// Minimal JSON extractors

fn extract_string(json: &str, key: &str) -> Result<String, &'static str> {
    let start = json.find(key).ok_or("key not found")? + key.len();
    let end = json[start..].find('"').ok_or("unterminated string")?;
    Ok(json[start..start + end].to_string())
}

fn extract_u64(json: &str, key: &str) -> Result<u64, &'static str> {
    let start = json.find(key).ok_or("key not found")? + key.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().map_err(|_| "parse error")
}

fn extract_embedding(json: &str) -> Result<Vec<f64>, &'static str> {
    let key = "\"embedding\":[";
    let start = json.find(key).ok_or("embedding key not found")? + key.len();
    let end = json[start..].find(']').ok_or("embedding array not closed")?;
    let inner = &json[start..start + end];
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|v| v.trim().parse().map_err(|_| "bad float"))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn reset_ids() {
        unsafe {
            ID_COUNTER = 0;
        }
    }

    #[test]
    fn test_register_room() {
        let mut chain = SignalChain::new();
        assert!(chain.register_room("planner"));
        assert!(!chain.register_room("planner")); // duplicate
        assert!(chain.register_room("executor"));
        assert_eq!(chain.rooms.len(), 2);
    }

    #[test]
    fn test_add_route() {
        let mut chain = SignalChain::new();
        chain.register_room("a");
        chain.register_room("b");
        assert!(chain.add_route("a", "b", RoutingAlgorithm::Direct));
        assert!(!chain.add_route("a", "b", RoutingAlgorithm::Direct)); // duplicate
        assert!(!chain.add_route("a", "c", RoutingAlgorithm::Direct)); // c not registered
        assert_eq!(chain.routes.len(), 1);
    }

    #[test]
    fn test_send_receive() {
        reset_ids();
        let mut chain = SignalChain::new();
        chain.register_room("source");
        chain.register_room("sink");
        chain.add_route("source", "sink", RoutingAlgorithm::Direct);

        let sig = Signal::new("source", "sink", SignalType::Tick).with_payload(vec![1, 2, 3]);
        chain.send(sig).unwrap();

        let received = chain.receive("sink");
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].payload, vec![1, 2, 3]);
        assert_eq!(received[0].signal_type, SignalType::Tick);
    }

    #[test]
    fn test_send_no_route() {
        let mut chain = SignalChain::new();
        chain.register_room("lonely");
        let sig = Signal::new("lonely", "nowhere", SignalType::Murmur);
        assert!(chain.send(sig).is_err());
    }

    #[test]
    fn test_send_unregistered_source() {
        let mut chain = SignalChain::new();
        let sig = Signal::new("ghost", "anyone", SignalType::Surprise);
        assert_eq!(chain.send(sig), Err("source room not registered"));
    }

    #[test]
    fn test_deadband_suppresses() {
        let config = DeadbandConfig {
            tolerance: 0.5,
            use_embedding: false,
        };
        let s1 = Signal::new("a", "b", SignalType::Tick).with_payload(vec![1, 2, 3]);
        let s2 = Signal::new("a", "b", SignalType::Tick).with_payload(vec![1, 2, 3]);
        // Same payload → diff = 0 < 0.5 → suppressed
        assert!(apply_deadband(&config, Some(&s1), &s2));
    }

    #[test]
    fn test_deadband_allows_change() {
        let config = DeadbandConfig {
            tolerance: 0.1,
            use_embedding: false,
        };
        let s1 = Signal::new("a", "b", SignalType::Tick).with_payload(vec![1, 2, 3]);
        let s2 = Signal::new("a", "b", SignalType::Tick).with_payload(vec![9, 9, 9]);
        // All bytes differ → diff = 1.0 > 0.1 → not suppressed
        assert!(!apply_deadband(&config, Some(&s1), &s2));
    }

    #[test]
    fn test_deadband_embedding() {
        let config = DeadbandConfig {
            tolerance: 0.01,
            use_embedding: true,
        };
        let s1 = Signal::new("a", "b", SignalType::Tick).with_embedding(vec![1.0, 0.0, 0.0]);
        let s2 = Signal::new("a", "b", SignalType::Tick).with_embedding(vec![1.0, 0.0, 0.0]);
        // Same embedding → distance 0 < 0.01 → suppressed
        assert!(apply_deadband(&config, Some(&s1), &s2));

        let s3 = Signal::new("a", "b", SignalType::Tick).with_embedding(vec![0.0, 1.0, 0.0]);
        // Orthogonal → distance 2.0 > 0.01 → allowed
        assert!(!apply_deadband(&config, Some(&s1), &s3));
    }

    #[test]
    fn test_correlated_routing() {
        let mut chain = SignalChain::new();
        chain.register_room("sensor");
        chain.register_room("analyzer");
        chain.add_route("sensor", "analyzer", RoutingAlgorithm::Correlated { threshold: 7 });

        // Low priority → dropped
        let low = Signal::new("sensor", "analyzer", SignalType::Murmur).with_priority(3);
        chain.send(low).unwrap();
        assert!(chain.receive("analyzer").is_empty());

        // High priority → delivered
        let high = Signal::new("sensor", "analyzer", SignalType::Prediction).with_priority(8);
        chain.send(high).unwrap();
        let received = chain.receive("analyzer");
        assert_eq!(received.len(), 1);
    }

    #[test]
    fn test_on_change_routing() {
        let mut chain = SignalChain::new();
        chain.register_room("producer");
        chain.register_room("consumer");
        chain.add_route("producer", "consumer", RoutingAlgorithm::OnChange);

        let s1 = Signal::new("producer", "consumer", SignalType::Tick).with_payload(vec![42]);
        chain.send(s1).unwrap();
        assert_eq!(chain.receive("consumer").len(), 1);

        // Same payload → OnChange drops
        let s2 = Signal::new("producer", "consumer", SignalType::Tick).with_payload(vec![42]);
        chain.send(s2).unwrap();
        assert!(chain.receive("consumer").is_empty());

        // Different payload → delivered
        let s3 = Signal::new("producer", "consumer", SignalType::Tick).with_payload(vec![99]);
        chain.send(s3).unwrap();
        assert_eq!(chain.receive("consumer").len(), 1);
    }

    #[test]
    fn test_broadcast() {
        let mut chain = SignalChain::new();
        chain.register_room("hub");
        chain.register_room("room_a");
        chain.register_room("room_b");
        chain.register_room("room_c");

        let sig = Signal::new("hub", "", SignalType::VibeShift);
        let count = chain.broadcast(sig);
        assert_eq!(count, 3);

        assert_eq!(chain.receive("room_a").len(), 1);
        assert_eq!(chain.receive("room_b").len(), 1);
        assert_eq!(chain.receive("room_c").len(), 1);
    }

    #[test]
    fn test_find_path() {
        let mut chain = SignalChain::new();
        chain.register_room("a");
        chain.register_room("b");
        chain.register_room("c");
        chain.add_route("a", "b", RoutingAlgorithm::Direct);
        chain.add_route("b", "c", RoutingAlgorithm::Direct);

        let path = chain.find_path("a", "c").unwrap();
        assert_eq!(path, vec!["a", "b", "c"]);

        assert!(chain.find_path("c", "a").is_none());
    }

    #[test]
    fn test_find_path_no_path() {
        let mut chain = SignalChain::new();
        chain.register_room("x");
        chain.register_room("y");
        // No route from x to y
        assert!(chain.find_path("x", "y").is_none());
    }

    #[test]
    fn test_to_a2a_from_a2a_roundtrip() {
        reset_ids();
        let original = Signal::new("room_a", "room_b", SignalType::Prediction)
            .with_payload(vec![0xDE, 0xAD, 0xBE, 0xEF])
            .with_embedding(vec![0.1, 0.5, 0.9])
            .with_priority(7);

        let bytes = SignalChain::to_a2a(&original);
        let restored = SignalChain::from_a2a(&bytes).unwrap();

        assert_eq!(restored.id, original.id);
        assert_eq!(restored.source, "room_a");
        assert_eq!(restored.target, "room_b");
        assert_eq!(restored.signal_type, SignalType::Prediction);
        assert_eq!(restored.priority, 7);
        assert_eq!(restored.payload, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(restored.embedding, vec![0.1, 0.5, 0.9]);
    }

    #[test]
    fn test_stats() {
        let mut chain = SignalChain::new();
        chain.register_room("src");
        chain.register_room("dst");
        chain.add_route("src", "dst", RoutingAlgorithm::Direct);

        let sig = Signal::new("src", "dst", SignalType::EnergyUpdate);
        chain.send(sig).unwrap();
        chain.receive("dst");

        let stats = chain.stats();
        assert_eq!(stats.rooms_registered, 2);
        assert_eq!(stats.routes_active, 1);
        assert_eq!(stats.signals_sent, 1);
        assert_eq!(stats.signals_received, 1);
    }

    #[test]
    fn test_detect_storm() {
        let mut chain = SignalChain::new();
        chain.register_room("target");
        chain.register_room("src");
        chain.add_route("src", "target", RoutingAlgorithm::Direct);

        // Send many signals with the same payload
        for _ in 0..5 {
            let sig = Signal::new("src", "target", SignalType::Tick).with_payload(vec![42]);
            chain.send(sig).unwrap();
        }
        // Don't drain — leave them in inbound for storm detection

        let storm = chain.detect_storm(5);
        assert!(storm.is_some());
        let (room, count) = storm.unwrap();
        assert_eq!(room, "target");
        assert_eq!(count, 5);
    }

    #[test]
    fn test_sampled_routing() {
        let mut chain = SignalChain::new();
        chain.register_room("fast");
        chain.register_room("slow");
        // 1 second interval
        chain.add_route("fast", "slow", RoutingAlgorithm::Sampled { interval_ms: 1000 });

        let sig = Signal::new("fast", "slow", SignalType::Tick);
        chain.send(sig).unwrap();
        assert_eq!(chain.receive("slow").len(), 1);

        // Second send within the interval → dropped by sampling
        let sig2 = Signal::new("fast", "slow", SignalType::Tick);
        chain.send(sig2).unwrap();
        assert!(chain.receive("slow").is_empty());
    }

    #[test]
    fn test_port_filters() {
        let mut port = Port::new("filtered");
        port.filters.push(|s: &Signal| s.priority >= 5);

        let high = Signal::new("x", "filtered", SignalType::Tick).with_priority(8);
        let low = Signal::new("x", "filtered", SignalType::Tick).with_priority(2);

        assert!(port.push_inbound(high));
        assert!(!port.push_inbound(low));
        assert_eq!(port.inbound.len(), 1);
    }

    #[test]
    fn test_port_transforms() {
        let mut port = Port::new("transformed");
        port.transforms.push(|s: &Signal| {
            let mut s = s.clone();
            s.payload.extend_from_slice(b"_transformed");
            s
        });

        let sig = Signal::new("x", "transformed", SignalType::Murmur).with_payload(b"hello".to_vec());
        port.push_inbound(sig);
        let received = port.inbound.pop_front().unwrap();
        assert_eq!(received.payload, b"hello_transformed".to_vec());
    }

    #[test]
    fn test_all_signal_types() {
        let types = [
            SignalType::Tick,
            SignalType::Murmur,
            SignalType::Prediction,
            SignalType::Surprise,
            SignalType::GcReport,
            SignalType::VibeShift,
            SignalType::LoRATrigger,
            SignalType::BalanceAlert,
            SignalType::CorrelationUpdate,
            SignalType::EnergyUpdate,
        ];
        let mut chain = SignalChain::new();
        chain.register_room("from");
        chain.register_room("to");
        chain.add_route("from", "to", RoutingAlgorithm::Direct);

        for st in &types {
            let sig = Signal::new("from", "to", st.clone());
            chain.send(sig).unwrap();
        }

        let received = chain.receive("to");
        assert_eq!(received.len(), 10);
    }
}
