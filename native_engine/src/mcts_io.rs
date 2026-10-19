use std::io::Write;
use std::path::Path;

use crate::mcts::PLANE_BYTES;

pub struct PositionRecord {
    pub planes: [u8; PLANE_BYTES],
    pub visits: Vec<(usize, u32)>,
}

pub struct GameRecord {
    pub positions: Vec<PositionRecord>,
    pub outcome: i8,
}

pub fn write_game_atomically(path: &Path, record: &GameRecord) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        let positions_json: Vec<String> = record
            .positions
            .iter()
            .map(|p| {
                let planes_json: Vec<String> = p.planes.iter().map(|b| b.to_string()).collect();
                let visits_json: Vec<String> = p
                    .visits
                    .iter()
                    .map(|(idx, visits)| format!("[{idx},{visits}]"))
                    .collect();
                format!(
                    "{{\"planes\":[{}],\"visits\":[{}]}}",
                    planes_json.join(","),
                    visits_json.join(",")
                )
            })
            .collect();
        write!(
            f,
            "{{\"outcome\":{},\"positions\":[{}]}}\n",
            record.outcome,
            positions_json.join(",")
        )?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn writes_valid_json_and_no_tmp_file_survives() {
        let dir = std::env::temp_dir().join("santoni_mcts_io_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("game-0001.json");
        let mut planes0 = [0u8; PLANE_BYTES];
        planes0[3] = 1;
        let mut planes1 = [0u8; PLANE_BYTES];
        planes1[149] = 1;
        let record = GameRecord {
            outcome: 1,
            positions: vec![
                PositionRecord { planes: planes0, visits: vec![(42, 100), (7, 88)] },
                PositionRecord { planes: planes1, visits: vec![(3, 12)] },
            ],
        };
        write_game_atomically(&path, &record).unwrap();
        let text = fs::read_to_string(&path).unwrap();

        let mut expected_planes0 = vec!["0"; PLANE_BYTES];
        expected_planes0[3] = "1";
        let mut expected_planes1 = vec!["0"; PLANE_BYTES];
        expected_planes1[149] = "1";
        let expected = format!(
            "{{\"outcome\":1,\"positions\":[{{\"planes\":[{}],\"visits\":[[42,100],[7,88]]}},{{\"planes\":[{}],\"visits\":[[3,12]]}}]}}\n",
            expected_planes0.join(","),
            expected_planes1.join(","),
        );
        assert_eq!(text, expected);
        assert!(!path.with_extension("json.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
