use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum Material {
    #[default]
    Air = 0,

    // Host rocks
    Sandstone = 1,
    Limestone = 2,
    Granite = 3,
    Basalt = 4,
    Slate = 5,
    Marble = 6,

    // Ores
    Iron = 7,
    Copper = 8,
    Malachite = 9,
    Tin = 10,
    Gold = 11,
    Diamond = 12,

    // Special formations
    Kimberlite = 13,
    Sulfide = 14,

    // Indicators / decorative
    Quartz = 15,
    Pyrite = 16,
    Amethyst = 17,
    Crystal = 18,

    // Sedimentary fuel
    Coal = 19,
}

impl Material {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Material::Sandstone,
            2 => Material::Limestone,
            3 => Material::Granite,
            4 => Material::Basalt,
            5 => Material::Slate,
            6 => Material::Marble,
            7 => Material::Iron,
            8 => Material::Copper,
            9 => Material::Malachite,
            10 => Material::Tin,
            11 => Material::Gold,
            12 => Material::Diamond,
            13 => Material::Kimberlite,
            14 => Material::Sulfide,
            15 => Material::Quartz,
            16 => Material::Pyrite,
            17 => Material::Amethyst,
            18 => Material::Crystal,
            19 => Material::Coal,
            _ => Material::Air,
        }
    }

    pub fn is_solid(self) -> bool {
        self != Material::Air
    }

    pub fn is_ore(self) -> bool {
        matches!(
            self,
            Material::Iron
                | Material::Copper
                | Material::Malachite
                | Material::Tin
                | Material::Gold
                | Material::Diamond
                | Material::Sulfide
                | Material::Coal
        )
    }

    pub fn is_carbonate(self) -> bool {
        matches!(self, Material::Limestone | Material::Marble)
    }

    pub fn is_host_rock(self) -> bool {
        matches!(
            self,
            Material::Sandstone
                | Material::Limestone
                | Material::Granite
                | Material::Basalt
                | Material::Slate
                | Material::Marble
        )
    }

    pub fn is_geode_shell(self) -> bool {
        matches!(self, Material::Crystal | Material::Amethyst)
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Material::Air => "Air",
            Material::Sandstone => "Sandstone",
            Material::Limestone => "Limestone",
            Material::Granite => "Granite",
            Material::Basalt => "Basalt",
            Material::Slate => "Slate",
            Material::Marble => "Marble",
            Material::Iron => "Iron",
            Material::Copper => "Copper",
            Material::Malachite => "Malachite",
            Material::Tin => "Tin",
            Material::Gold => "Gold",
            Material::Diamond => "Diamond",
            Material::Kimberlite => "Kimberlite",
            Material::Sulfide => "Sulfide",
            Material::Quartz => "Quartz",
            Material::Pyrite => "Pyrite",
            Material::Amethyst => "Amethyst",
            Material::Crystal => "Crystal",
            Material::Coal => "Coal",
        }
    }

    pub fn color_hex(self) -> u32 {
        match self {
            Material::Air => 0x000000,
            Material::Sandstone => 0xC2A366,
            Material::Limestone => 0xD4C4A8,
            Material::Granite => 0x8B7D6B,
            Material::Basalt => 0x3B3B3B,
            Material::Slate => 0x5A6B7A,
            Material::Marble => 0xE8E0D8,
            Material::Iron => 0xA0522D,
            Material::Copper => 0xB87333,
            Material::Malachite => 0x0BDA51,
            Material::Tin => 0xC0C0C0,
            Material::Gold => 0xFFD700,
            Material::Diamond => 0xB9F2FF,
            Material::Kimberlite => 0x4A3728,
            Material::Sulfide => 0x8B8000,
            Material::Quartz => 0xE8E0D0,
            Material::Pyrite => 0xCBA135,
            Material::Amethyst => 0x9B59B6,
            Material::Crystal => 0x85C1E9,
            Material::Coal => 0x2C2C2C,
        }
    }

    /// All solid material variants (for palette generation).
    pub fn all_solid() -> &'static [Material] {
        &[
            Material::Sandstone,
            Material::Limestone,
            Material::Granite,
            Material::Basalt,
            Material::Slate,
            Material::Marble,
            Material::Iron,
            Material::Copper,
            Material::Malachite,
            Material::Tin,
            Material::Gold,
            Material::Diamond,
            Material::Kimberlite,
            Material::Sulfide,
            Material::Quartz,
            Material::Pyrite,
            Material::Amethyst,
            Material::Crystal,
            Material::Coal,
        ]
    }
}
