use std::io::Write;
use std::path::Path;

pub struct GameRecord {
    pub moves: Vec<(usize, u32)>,
    pub outcome: i8,
}

pub fn write_game_atomically(path: &Path, record: &GameRecord) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        let moves_json: Vec<String> = record
            .moves
            .iter()
            .map(|(idx, visits)| format!("[{idx},{visits}]"))
            .collect();
        write!(
            f,
            "{{\"outcome\":{},\"moves\":[{}]}}\n",
            record.outcome,
            moves_json.join(",")
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
        let record = GameRecord { moves: vec![(42, 100), (7, 88)], outcome: 1 };
        write_game_atomically(&path, &record).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(text, "{\"outcome\":1,\"moves\":[[42,100],[7,88]]}\n");
        assert!(!path.with_extension("json.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
