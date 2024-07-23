//! Angled Random Walker approach to Brownian tree generation.
//!
//! Lets you generate Brownian-tree-like structures faster than traditional
//! diffusion-limited aggregation approaches.
//!
//! Useful for procedurally generating heightmaps.

use rand::distributions::{Distribution, WeightedIndex};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[derive(Clone, Copy)]
struct Direction(i32, i32);

const DIRECTIONS: [Direction; 8] = [
    Direction(1, 0),   // East
    Direction(1, -1),  // North-East
    Direction(0, -1),  // North
    Direction(-1, -1), // North-West
    Direction(-1, 0),  // West
    Direction(-1, 1),  // South-West
    Direction(0, 1),   // South
    Direction(1, 1),   // South-East
];

impl Direction {
    fn to_angle(self) -> f64 {
        match self {
            Direction(1, 0) => 0.00,
            Direction(1, -1) => 0.25,
            Direction(0, -1) => 0.50,
            Direction(-1, -1) => 0.75,
            Direction(-1, 0) => 1.00,
            Direction(-1, 1) => 1.25,
            Direction(0, 1) => 1.50,
            Direction(1, 1) => 1.75,
            Direction(x, y) => {
                (f64::atan2(y as f64, x as f64) - f64::atan2(0.00, 1.00)) / std::f64::consts::PI
            }
        }
    }
}

fn angle_distance(a: f64, b: f64) -> f64 {
    f64::min((a - b).abs(), 2.00 - (a - b).abs())
}

fn angle_to_direction_weights(angle: f64) -> [f64; 8] {
    let weights = DIRECTIONS.map(|d| 2.00 - angle_distance(angle, d.to_angle()));
    let least_likely = weights.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    weights.map(|w| w - least_likely)
}

fn choose_direction(angle: f64, rng: &mut Xoshiro256PlusPlus) -> Direction {
    let dist = WeightedIndex::new(angle_to_direction_weights(angle)).unwrap();
    DIRECTIONS[dist.sample(rng)]
}

fn angle_displace_random(angle: f64, divergence: f64, rng: &mut Xoshiro256PlusPlus) -> f64 {
    (angle + rng.gen_range(-divergence..divergence)).clamp(0.00, 2.00)
}

#[derive(Clone, Copy, Debug)]
struct Position(usize, usize);

impl Position {
    fn move_in(&self, dir: Direction, size: usize) -> Self {
        let x = ((self.0 as i32) + dir.0).clamp(0, (size - 1) as i32) as usize;
        let y = ((self.1 as i32) + dir.1).clamp(0, (size - 1) as i32) as usize;
        Self(x, y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum WalkerKind {
    Short,
    Long,
}

#[derive(Clone, Copy, Debug)]
struct AngledRandomWalker {
    kind: WalkerKind,
    age: usize,
    cumulative_age: usize,
    generation: usize,
    position: Position,
    angle: f64,
}

/// What values should be used to fill the grid.
pub enum Paint {
    /// The age of the walker when it reached that point.
    Age,
    /// The cumulative age, i.e. the age of the walker plus that of all its ancestors.
    CumulativeAge,
    /// The walker's generation (older is lower).
    Generation,
    /// The constant value `1`.
    Constant,
}

/// Determines the orientation of the initial walkers placed at the start of the simulation.
pub enum InitialWalkers {
    /// Eight walkers, oriented towards each one of the cardinal and ordinal directions (N, S, E, W, NE, SE, NW, SW).
    CardinalsAndOrdinals,
    /// Walkers will be oriented at the angles in this vector. Angles are in units of Pi radians. Values will be clamped to `0.00..=2.00`.
    Custom(Vec<f64>),
}

/// The parameters that control the simulation.
pub struct SimulationParams {
    /// Size of the grid.
    pub size: usize,
    /// Maximum age that long walkers survive to.
    pub max_long_age: usize,
    /// Maximum age that short walkers survive to.
    pub max_short_age: usize,
    /// Maximum generations of walkers.
    pub max_generations: usize,
    /// How many children a walker spawns when it ends.
    pub children: usize,
    /// Maximum angle that a long walker can diverge from its parent. Angles are in units of Pi radians, and clamped to `0.00..=2.00`.
    pub max_long_angle_divergence: f64,
    /// Maximum angle that a short walker can diverge from its parent. Angles are in units of Pi radians, and clamped to `0.00..=2.00`.
    pub max_short_angle_divergence: f64,
    /// How often a long walker produces a short branch.
    pub short_branch_frequency: usize,
    /// What to fill the grid with.
    pub paint: Paint,
    /// How to place initial walkers.
    pub initial_walkers: InitialWalkers,
    /// Seed for the randomness.
    pub seed: u64,
}

/// Main simulation function.
pub fn simulate(params: SimulationParams) -> Vec<Vec<u8>> {
    let mut grid = vec![vec![0u8; params.size]; params.size];
    let mut walkers: Vec<AngledRandomWalker> = match params.initial_walkers {
        InitialWalkers::CardinalsAndOrdinals => (0..8)
            .map(|n| AngledRandomWalker {
                kind: WalkerKind::Long,
                age: 0,
                cumulative_age: 0,
                generation: 0,
                position: Position(params.size / 2, params.size / 2),
                angle: 0.25 * n as f64,
            })
            .collect(),
        InitialWalkers::Custom(angles) => angles
            .into_iter()
            .map(|angle| AngledRandomWalker {
                kind: WalkerKind::Long,
                age: 0,
                cumulative_age: 0,
                generation: 0,
                position: Position(params.size / 2, params.size / 2),
                angle: angle.clamp(0.00, 2.00),
            })
            .collect(),
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(params.seed);
    while !walkers.is_empty() {
        let mut next_walkers: Vec<AngledRandomWalker> = Vec::new();
        for walker in &walkers {
            let max_age = match walker.kind {
                WalkerKind::Short => params.max_short_age,
                WalkerKind::Long => params.max_long_age,
            };
            if walker.age % params.short_branch_frequency == 0
                && walker.kind != WalkerKind::Short
                && walker.generation < params.max_generations
            {
                next_walkers.push(AngledRandomWalker {
                    kind: WalkerKind::Short,
                    age: 0,
                    cumulative_age: walker.cumulative_age + walker.age,
                    generation: walker.generation + 1,
                    position: walker.position,
                    angle: angle_displace_random(
                        walker.angle,
                        params.max_short_angle_divergence.clamp(0.00, 2.00),
                        &mut rng,
                    ),
                })
            }
            if walker.age > max_age {
                if walker.generation < params.max_generations && walker.kind != WalkerKind::Short {
                    for _ in 0..params.children {
                        next_walkers.push(AngledRandomWalker {
                            kind: WalkerKind::Long,
                            age: 0,
                            cumulative_age: walker.cumulative_age + walker.age,
                            generation: walker.generation + 1,
                            position: walker.position,
                            angle: angle_displace_random(
                                walker.angle,
                                params.max_long_angle_divergence.clamp(0.00, 2.00),
                                &mut rng,
                            ),
                        });
                    }
                }
            } else {
                let Position(x, y) = walker
                    .position
                    .move_in(choose_direction(walker.angle, &mut rng), params.size);

                grid[y][x] = match params.paint {
                    Paint::Age => walker.age + 1,
                    Paint::CumulativeAge => walker.cumulative_age + walker.age + 1,
                    Paint::Generation => walker.generation + 1,
                    Paint::Constant => 1,
                } as u8;
                next_walkers.push(AngledRandomWalker {
                    kind: walker.kind,
                    age: walker.age + 1,
                    cumulative_age: walker.cumulative_age,
                    generation: walker.generation,
                    position: Position(x, y),
                    angle: walker.angle,
                });
            }
        }
        walkers = next_walkers;
    }
    grid
}
