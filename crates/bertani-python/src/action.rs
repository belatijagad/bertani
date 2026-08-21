//! Stable tensor action IDs and conversion to the rules-engine action types.
//!
//! The tensor layout is row-major `[player, slot, (op, arg, count)]`.  Unit
//! slot zero is the farmer and subsequent slots are farm hands.  Market rows
//! retain their slot positions because Kaggriculture resolves the two players'
//! orders in lockstep by slot.

use std::{error::Error, fmt};

use kaggriculture_core::{Action, Crop, Item, MarketOrder, UnitAction};

pub const PLAYER_COUNT: usize = 2;
pub const ACTION_FIELD_COUNT: usize = 3;

pub const UNIT_ACTION_COUNT: usize = 18;
pub const UNIT_PASS: i64 = 0;
pub const UNIT_NORTH: i64 = 1;
pub const UNIT_SOUTH: i64 = 2;
pub const UNIT_EAST: i64 = 3;
pub const UNIT_WEST: i64 = 4;
pub const UNIT_PICKUP: i64 = 5;
pub const UNIT_DROP: i64 = 6;
pub const UNIT_PLACE: i64 = 7;
pub const UNIT_PLANT: i64 = 8;
pub const UNIT_WATER: i64 = 9;
pub const UNIT_HARVEST: i64 = 10;
pub const UNIT_FERTILIZE: i64 = 11;
pub const UNIT_DIG: i64 = 12;
pub const UNIT_BUILD_COOP: i64 = 13;
pub const UNIT_BUILD_PASTURE: i64 = 14;
pub const UNIT_FEED: i64 = 15;
pub const UNIT_COLLECT_FERTILIZER: i64 = 16;
pub const UNIT_CARE: i64 = 17;

pub const MARKET_ACTION_COUNT: usize = 7;
pub const MARKET_NONE: i64 = 0;
pub const MARKET_HIRE: i64 = 1;
pub const MARKET_BUY_LAND: i64 = 2;
pub const MARKET_BUY_SEED: i64 = 3;
pub const MARKET_BUY_PRODUCT: i64 = 4;
pub const MARKET_BUY_ANIMAL: i64 = 5;
pub const MARKET_SELL: i64 = 6;

pub const ITEM_COUNT: usize = 12;
pub const ITEM_WHEAT: i64 = 0;
pub const ITEM_CARROT: i64 = 1;
pub const ITEM_TOMATO: i64 = 2;
pub const ITEM_STRAWBERRY: i64 = 3;
pub const ITEM_MELON: i64 = 4;
pub const ITEM_EGG: i64 = 5;
pub const ITEM_MILK: i64 = 6;
pub const ITEM_WOOL: i64 = 7;
pub const ITEM_FERTILIZER: i64 = 8;
pub const ITEM_GOOSE: i64 = 9;
pub const ITEM_COW: i64 = 10;
pub const ITEM_SHEEP: i64 = 11;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DecodeError {
    MaxUnitsIsZero,
    MarketPrefixTooLong {
        player: usize,
        market_len: usize,
        max_orders: usize,
    },
    CapacityOverflow {
        buffer: &'static str,
    },
    BufferLength {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidOperation {
        kind: &'static str,
        player: usize,
        slot: usize,
        value: i64,
        action_count: usize,
    },
    InvalidArgument {
        kind: &'static str,
        player: usize,
        slot: usize,
        operation: i64,
        value: i64,
        expected: &'static str,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxUnitsIsZero => write!(
                formatter,
                "max_units must include the farmer and be at least 1"
            ),
            Self::MarketPrefixTooLong {
                player,
                market_len,
                max_orders,
            } => write!(
                formatter,
                "market length {market_len} for player {player} exceeds max_orders ({max_orders})"
            ),
            Self::CapacityOverflow { buffer } => {
                write!(formatter, "{buffer} tensor dimensions overflow usize")
            }
            Self::BufferLength {
                buffer,
                expected,
                actual,
            } => write!(
                formatter,
                "{buffer} buffer has {actual} values; expected exactly {expected}"
            ),
            Self::InvalidOperation {
                kind,
                player,
                slot,
                value,
                action_count,
            } => write!(
                formatter,
                "invalid {kind} op {value} at player {player}, slot {slot}; expected 0..{action_count}"
            ),
            Self::InvalidArgument {
                kind,
                player,
                slot,
                operation,
                value,
                expected,
            } => write!(
                formatter,
                "invalid {kind} arg {value} for op {operation} at player {player}, slot {slot}; expected {expected}"
            ),
        }
    }
}

impl Error for DecodeError {}

/// Decode one vector-environment action tensor into the core's two actions.
///
/// `unit_rows` must have the flattened shape `[2, max_units, 3]` and
/// `market_rows` the shape `[2, max_orders, 3]`. Only represented farm hands
/// (up to `max_units - 1`) are read; unused unit rows and market rows after
/// that player's `market_lens` entry are deliberately ignored. Counts accept
/// the entire `i64`
/// domain. The core treats non-positive quantities as no-ops.
pub(crate) fn decode_actions(
    unit_rows: &[i64],
    market_rows: &[i64],
    max_units: usize,
    max_orders: usize,
    hand_counts: [usize; PLAYER_COUNT],
    market_lens: [usize; PLAYER_COUNT],
) -> Result<[Action; PLAYER_COUNT], DecodeError> {
    if max_units == 0 {
        return Err(DecodeError::MaxUnitsIsZero);
    }
    for (player, market_len) in market_lens.into_iter().enumerate() {
        if market_len > max_orders {
            return Err(DecodeError::MarketPrefixTooLong {
                player,
                market_len,
                max_orders,
            });
        }
    }

    validate_buffer_len("unit action", unit_rows, max_units)?;
    validate_buffer_len("market action", market_rows, max_orders)?;

    Ok([
        decode_player(
            unit_rows,
            market_rows,
            max_units,
            max_orders,
            hand_counts[0],
            market_lens[0],
            0,
        )?,
        decode_player(
            unit_rows,
            market_rows,
            max_units,
            max_orders,
            hand_counts[1],
            market_lens[1],
            1,
        )?,
    ])
}

fn decode_player(
    unit_rows: &[i64],
    market_rows: &[i64],
    max_units: usize,
    max_orders: usize,
    hand_count: usize,
    market_len: usize,
    player: usize,
) -> Result<Action, DecodeError> {
    let unit_base = player * max_units * ACTION_FIELD_COUNT;
    let farmer = decode_unit(row(unit_rows, unit_base), player, 0)?;
    let represented_hands = hand_count.min(max_units - 1);
    let mut hands = Vec::with_capacity(represented_hands);
    for hand in 0..represented_hands {
        let slot = hand + 1;
        hands.push(decode_unit(
            row(unit_rows, unit_base + slot * ACTION_FIELD_COUNT),
            player,
            slot,
        )?);
    }

    let market_base = player * max_orders * ACTION_FIELD_COUNT;
    let mut market = Vec::with_capacity(market_len);
    let mut remaining_hires = max_units.saturating_sub(1).saturating_sub(hand_count);
    for slot in 0..market_len {
        let order = decode_market(
            row(market_rows, market_base + slot * ACTION_FIELD_COUNT),
            player,
            slot,
        )?;
        if order == MarketOrder::Hire {
            if remaining_hires == 0 {
                // Preserve market-slot alignment while preventing the fixed
                // worker tensor from overflowing on the next observation.
                market.push(MarketOrder::Sell {
                    item: Item::Wheat,
                    count: 0,
                });
                continue;
            }
            remaining_hires -= 1;
        }
        market.push(order);
    }

    Ok(Action {
        farmer,
        hands,
        market,
    })
}

fn validate_buffer_len(
    buffer: &'static str,
    values: &[i64],
    slots_per_player: usize,
) -> Result<(), DecodeError> {
    let expected = PLAYER_COUNT
        .checked_mul(slots_per_player)
        .and_then(|size| size.checked_mul(ACTION_FIELD_COUNT))
        .ok_or(DecodeError::CapacityOverflow { buffer })?;
    if values.len() != expected {
        return Err(DecodeError::BufferLength {
            buffer,
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}

fn row(values: &[i64], offset: usize) -> [i64; ACTION_FIELD_COUNT] {
    [values[offset], values[offset + 1], values[offset + 2]]
}

fn decode_unit(
    [operation, argument, count]: [i64; ACTION_FIELD_COUNT],
    player: usize,
    slot: usize,
) -> Result<UnitAction, DecodeError> {
    if !(UNIT_PASS..i64::try_from(UNIT_ACTION_COUNT).expect("action count fits i64"))
        .contains(&operation)
    {
        return Err(DecodeError::InvalidOperation {
            kind: "unit",
            player,
            slot,
            value: operation,
            action_count: UNIT_ACTION_COUNT,
        });
    }

    let action = match operation {
        UNIT_PASS => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::Pass
        }
        UNIT_NORTH => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::North
        }
        UNIT_SOUTH => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::South
        }
        UNIT_EAST => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::East
        }
        UNIT_WEST => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::West
        }
        UNIT_PICKUP => UnitAction::Pickup {
            item: decode_item("unit", player, slot, operation, argument)?,
            count,
        },
        UNIT_DROP => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::Drop
        }
        UNIT_PLACE => UnitAction::Place {
            item: decode_item("unit", player, slot, operation, argument)?,
            count,
        },
        UNIT_PLANT => UnitAction::Plant(decode_crop("unit", player, slot, operation, argument)?),
        UNIT_WATER => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::Water
        }
        UNIT_HARVEST => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::Harvest
        }
        UNIT_FERTILIZE => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::Fertilize
        }
        UNIT_DIG => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::Dig
        }
        UNIT_BUILD_COOP => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::BuildCoop
        }
        UNIT_BUILD_PASTURE => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::BuildPasture
        }
        UNIT_FEED => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::Feed
        }
        UNIT_COLLECT_FERTILIZER => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::CollectFertilizer
        }
        UNIT_CARE => {
            expect_zero_argument("unit", player, slot, operation, argument)?;
            UnitAction::Care
        }
        _ => unreachable!("operation range checked above"),
    };
    Ok(action)
}

fn decode_market(
    [operation, argument, count]: [i64; ACTION_FIELD_COUNT],
    player: usize,
    slot: usize,
) -> Result<MarketOrder, DecodeError> {
    if !(MARKET_NONE..i64::try_from(MARKET_ACTION_COUNT).expect("action count fits i64"))
        .contains(&operation)
    {
        return Err(DecodeError::InvalidOperation {
            kind: "market",
            player,
            slot,
            value: operation,
            action_count: MARKET_ACTION_COUNT,
        });
    }

    let order = match operation {
        MARKET_NONE => {
            expect_zero_argument("market", player, slot, operation, argument)?;
            // There is intentionally no `MarketOrder::None`: a zero-quantity
            // sale survives in the Vec without changing the game, preserving
            // alignment with the opponent's order in the same market slot.
            MarketOrder::Sell {
                item: Item::Wheat,
                count: 0,
            }
        }
        MARKET_HIRE => {
            expect_zero_argument("market", player, slot, operation, argument)?;
            MarketOrder::Hire
        }
        MARKET_BUY_LAND => {
            expect_zero_argument("market", player, slot, operation, argument)?;
            MarketOrder::BuyLand
        }
        MARKET_BUY_SEED => MarketOrder::BuySeed {
            crop: decode_crop("market", player, slot, operation, argument)?,
            count,
        },
        MARKET_BUY_PRODUCT => {
            let item = decode_item("market", player, slot, operation, argument)?;
            if !matches!(item, Item::Wheat | Item::Fertilizer) {
                return Err(invalid_argument(
                    "market",
                    player,
                    slot,
                    operation,
                    argument,
                    "WHEAT (0) or FERTILIZER (8)",
                ));
            }
            MarketOrder::BuyProduct { item, count }
        }
        MARKET_BUY_ANIMAL => MarketOrder::BuyAnimal {
            animal: decode_item("market", player, slot, operation, argument)?
                .as_animal()
                .ok_or_else(|| {
                    invalid_argument(
                        "market",
                        player,
                        slot,
                        operation,
                        argument,
                        "an animal item ID in 9..12",
                    )
                })?,
            count,
        },
        MARKET_SELL => {
            let item = decode_item("market", player, slot, operation, argument)?;
            if item.as_product().is_none() {
                return Err(invalid_argument(
                    "market",
                    player,
                    slot,
                    operation,
                    argument,
                    "a product item ID in 0..9",
                ));
            }
            MarketOrder::Sell { item, count }
        }
        _ => unreachable!("operation range checked above"),
    };
    Ok(order)
}

fn expect_zero_argument(
    kind: &'static str,
    player: usize,
    slot: usize,
    operation: i64,
    argument: i64,
) -> Result<(), DecodeError> {
    if argument != 0 {
        return Err(invalid_argument(
            kind,
            player,
            slot,
            operation,
            argument,
            "0 (this operation has no argument)",
        ));
    }
    Ok(())
}

fn decode_item(
    kind: &'static str,
    player: usize,
    slot: usize,
    operation: i64,
    argument: i64,
) -> Result<Item, DecodeError> {
    usize::try_from(argument)
        .ok()
        .and_then(|index| Item::ALL.get(index).copied())
        .ok_or_else(|| {
            invalid_argument(
                kind,
                player,
                slot,
                operation,
                argument,
                "an item ID in 0..12",
            )
        })
}

fn decode_crop(
    kind: &'static str,
    player: usize,
    slot: usize,
    operation: i64,
    argument: i64,
) -> Result<Crop, DecodeError> {
    usize::try_from(argument)
        .ok()
        .and_then(|index| Crop::ALL.get(index).copied())
        .ok_or_else(|| {
            invalid_argument(
                kind,
                player,
                slot,
                operation,
                argument,
                "a crop item ID in 0..5",
            )
        })
}

fn invalid_argument(
    kind: &'static str,
    player: usize,
    slot: usize,
    operation: i64,
    value: i64,
    expected: &'static str,
) -> DecodeError {
    DecodeError::InvalidArgument {
        kind,
        player,
        slot,
        operation,
        value,
        expected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kaggriculture_core::Animal;

    fn zero_actions(max_units: usize, max_orders: usize) -> (Vec<i64>, Vec<i64>) {
        (
            vec![0; PLAYER_COUNT * max_units * ACTION_FIELD_COUNT],
            vec![0; PLAYER_COUNT * max_orders * ACTION_FIELD_COUNT],
        )
    }

    fn set_row(values: &mut [i64], slots: usize, player: usize, slot: usize, row: [i64; 3]) {
        let offset = (player * slots + slot) * ACTION_FIELD_COUNT;
        values[offset..offset + ACTION_FIELD_COUNT].copy_from_slice(&row);
    }

    #[test]
    fn decodes_every_unit_operation_and_preserves_counts() {
        let max_units = UNIT_ACTION_COUNT;
        let (mut units, market) = zero_actions(max_units, 0);
        for operation in 0..i64::try_from(UNIT_ACTION_COUNT).unwrap() {
            let argument = match operation {
                UNIT_PICKUP | UNIT_PLACE => ITEM_SHEEP,
                UNIT_PLANT => ITEM_MELON,
                _ => 0,
            };
            set_row(
                &mut units,
                max_units,
                0,
                usize::try_from(operation).unwrap(),
                [operation, argument, -7],
            );
        }

        let decoded = decode_actions(
            &units,
            &market,
            max_units,
            0,
            [UNIT_ACTION_COUNT - 1, 0],
            [0, 0],
        )
        .unwrap();
        let actual = std::iter::once(decoded[0].farmer)
            .chain(decoded[0].hands.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                UnitAction::Pass,
                UnitAction::North,
                UnitAction::South,
                UnitAction::East,
                UnitAction::West,
                UnitAction::Pickup {
                    item: Item::Sheep,
                    count: -7,
                },
                UnitAction::Drop,
                UnitAction::Place {
                    item: Item::Sheep,
                    count: -7,
                },
                UnitAction::Plant(Crop::Melon),
                UnitAction::Water,
                UnitAction::Harvest,
                UnitAction::Fertilize,
                UnitAction::Dig,
                UnitAction::BuildCoop,
                UnitAction::BuildPasture,
                UnitAction::Feed,
                UnitAction::CollectFertilizer,
                UnitAction::Care,
            ]
        );
    }

    #[test]
    fn decodes_every_market_operation_without_collapsing_none_gap() {
        let max_orders = MARKET_ACTION_COUNT;
        let (units, mut market) = zero_actions(2, max_orders);
        let rows = [
            [MARKET_NONE, 0, 123],
            [MARKET_HIRE, 0, -1],
            [MARKET_BUY_LAND, 0, 0],
            [MARKET_BUY_SEED, ITEM_STRAWBERRY, -3],
            [MARKET_BUY_PRODUCT, ITEM_FERTILIZER, 4],
            [MARKET_BUY_ANIMAL, ITEM_COW, 5],
            [MARKET_SELL, ITEM_WOOL, 6],
        ];
        for (slot, encoded) in rows.into_iter().enumerate() {
            set_row(&mut market, max_orders, 0, slot, encoded);
        }

        let decoded =
            decode_actions(&units, &market, 2, max_orders, [0, 0], [max_orders, 0]).unwrap();
        assert_eq!(
            decoded[0].market,
            vec![
                MarketOrder::Sell {
                    item: Item::Wheat,
                    count: 0,
                },
                MarketOrder::Hire,
                MarketOrder::BuyLand,
                MarketOrder::BuySeed {
                    crop: Crop::Strawberry,
                    count: -3,
                },
                MarketOrder::BuyProduct {
                    item: Item::Fertilizer,
                    count: 4,
                },
                MarketOrder::BuyAnimal {
                    animal: Animal::Cow,
                    count: 5,
                },
                MarketOrder::Sell {
                    item: Item::Wool,
                    count: 6,
                },
            ]
        );
    }

    #[test]
    fn ignores_inactive_rows_and_truncates_unrepresented_hands() {
        let (mut units, mut market) = zero_actions(2, 3);
        set_row(&mut units, 2, 1, 1, [i64::MAX, i64::MAX, i64::MAX]);
        set_row(&mut market, 3, 0, 2, [i64::MAX, i64::MAX, i64::MAX]);

        let decoded = decode_actions(&units, &market, 2, 3, [99, 0], [1, 2]).unwrap();
        assert_eq!(decoded[0].hands, vec![UnitAction::Pass]);
        assert_eq!(decoded[0].market.len(), 1);
        assert_eq!(decoded[1].market.len(), 2);
    }

    #[test]
    fn validates_shapes_prefix_and_active_fields() {
        let (mut units, mut market) = zero_actions(2, 1);
        assert!(matches!(
            decode_actions(&units[..units.len() - 1], &market, 2, 1, [0, 0], [0, 0]),
            Err(DecodeError::BufferLength {
                buffer: "unit action",
                ..
            })
        ));
        assert!(matches!(
            decode_actions(&units, &market, 0, 1, [0, 0], [0, 0]),
            Err(DecodeError::MaxUnitsIsZero)
        ));
        assert!(matches!(
            decode_actions(&units, &market, 2, 1, [0, 0], [0, 2]),
            Err(DecodeError::MarketPrefixTooLong { player: 1, .. })
        ));

        set_row(
            &mut units,
            2,
            1,
            0,
            [i64::try_from(UNIT_ACTION_COUNT).unwrap(), 0, 0],
        );
        let error = decode_actions(&units, &market, 2, 1, [0, 0], [1, 1]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid unit op 18 at player 1, slot 0; expected 0..18"
        );

        set_row(&mut units, 2, 1, 0, [UNIT_PASS, 0, 0]);
        set_row(&mut market, 1, 0, 0, [MARKET_BUY_ANIMAL, ITEM_MILK, 1]);
        let error = decode_actions(&units, &market, 2, 1, [0, 0], [1, 1]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid market arg 6 for op 5 at player 0, slot 0; expected an animal item ID in 9..12"
        );
    }

    #[test]
    fn numeric_ids_match_the_core_enum_order_contract() {
        assert_eq!(UNIT_ACTION_COUNT, 18);
        assert_eq!(MARKET_ACTION_COUNT, 7);
        assert_eq!(ITEM_COUNT, Item::COUNT);
        assert_eq!(ITEM_WHEAT, i64::try_from(Item::Wheat.index()).unwrap());
        assert_eq!(ITEM_CARROT, i64::try_from(Item::Carrot.index()).unwrap());
        assert_eq!(ITEM_TOMATO, i64::try_from(Item::Tomato.index()).unwrap());
        assert_eq!(
            ITEM_STRAWBERRY,
            i64::try_from(Item::Strawberry.index()).unwrap()
        );
        assert_eq!(ITEM_MELON, i64::try_from(Item::Melon.index()).unwrap());
        assert_eq!(ITEM_EGG, i64::try_from(Item::Egg.index()).unwrap());
        assert_eq!(ITEM_MILK, i64::try_from(Item::Milk.index()).unwrap());
        assert_eq!(ITEM_WOOL, i64::try_from(Item::Wool.index()).unwrap());
        assert_eq!(
            ITEM_FERTILIZER,
            i64::try_from(Item::Fertilizer.index()).unwrap()
        );
        assert_eq!(ITEM_GOOSE, i64::try_from(Item::Goose.index()).unwrap());
        assert_eq!(ITEM_COW, i64::try_from(Item::Cow.index()).unwrap());
        assert_eq!(ITEM_SHEEP, i64::try_from(Item::Sheep.index()).unwrap());
    }
}
