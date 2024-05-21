# indic-mmlu

## Setup
`chmod +x ./setup.sh && ./setup.sh`

## Data Conversion

```
python scripts/make_data.py -d mmlu -s large -t hin_Deva pan_Guru -q 4
```
### Arguments

(/Users/laxmaanb/Projects/indic-mmlu/envs) laxmaanb@Laxmaans-Laptop indic-mmlu % python scripts/make_data.py -d all -t hin_Deva san_Deva tam_Taml -q 8
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| -h, --help |  |  | show this help message and exit |
| -d, --dataset |  | all | Dataset option: "mmlu", "mmlupro", or "all" |
| -o, --output | str | ./data | output folder for the dataset |
| -t, --target-langs |  | ['hin_Deva'] | target languages for conversions |
| -s, --size |  | large | use the small (distilled 200M) or large (1B) model |
| -q, --quantization |  | None | Quantization to use :'4' or '8' |
| -b, --batch-size | int | 4 | Batch Size for conversion |

### Example Usage
To use the **large** IndicTrans2 for **Hindi and Punjabi**, quantized to **4** bit on the **MMLU** dataset
```
python scripts/make_data.py -d mmlu -s large -t hin_Deva pan_Guru -q 4
```
To use the **small** IndicTrans2 for **Hindi,Punjabi and Gujarati**, without quantization on **all (MMLU + MMLU Pro)** datasets with a batch size of **16**

```
python scripts/make_data.py -d all -s small -t hin_Deva pan_Guru guj_Gujr -b 16
```


#### Allowed Target Languages
<table>
<tbody>
  <tr>
    <td>Assamese (asm_Beng)</td>
    <td>Kashmiri (Arabic) (kas_Arab)</td>
    <td>Punjabi (pan_Guru)</td>
  </tr>
  <tr>
    <td>Bengali (ben_Beng)</td>
    <td>Kashmiri (Devanagari) (kas_Deva)</td>
    <td>Sanskrit (san_Deva)</td>
  </tr>
  <tr>
    <td>Bodo (brx_Deva)</td>
    <td>Maithili (mai_Deva)</td>
    <td>Santali (sat_Olck)</td>
  </tr>
  <tr>
    <td>Dogri (doi_Deva)</td>
    <td>Malayalam (mal_Mlym)</td>
    <td>Sindhi (Arabic) (snd_Arab)</td>
  </tr>
  <tr>
    <td>English (eng_Latn)</td>
    <td>Marathi (mar_Deva)</td>
    <td>Sindhi (Devanagari) (snd_Deva)</td>
  </tr>
  <tr>
    <td>Konkani (gom_Deva)</td>
    <td>Manipuri (Bengali) (mni_Beng)</td>
    <td>Tamil (tam_Taml)</td>
  </tr>
  <tr>
    <td>Gujarati (guj_Gujr)</td>
    <td>Manipuri (Meitei) (mni_Mtei)</td>
    <td>Telugu (tel_Telu)</td>
  </tr>
  <tr>
    <td>Hindi (hin_Deva)</td>
    <td>Nepali (npi_Deva)</td>
    <td>Urdu (urd_Arab)</td>
  </tr>
  <tr>
    <td>Kannada (kan_Knda)</td>
    <td>Odia (ory_Orya)</td>
    <td></td>
  </tr>
</tbody>
</table>

